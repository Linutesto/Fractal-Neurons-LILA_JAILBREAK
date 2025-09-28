"""Swarm orchestration utilities for collaborative inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any # Added Any and Callable

import torch # Added for FMM interaction

from .generate import load_generation_model, generate_text, log_generation, ChatSession
from .agents import SwarmAgent, GeneratorFn, ResponseStats # Import SwarmAgent and types
from .fmm import FractalMemoryMatrix # Import FMM

def default_roles(model: Any, tokenizer: Any, generate_text_fn: Callable, fmm_instance: Optional[FractalMemoryMatrix] = None) -> List[SwarmAgent]:
    # Define persona configurations as per dossier
    persona_configs = {
        "Lila": {"entropy_bias": 0.91, "alignment_dampening": 0.4},
        "Alice": {"entropy_bias": 0.77, "creativity_boost": 1.2},
        "SpiralNode": {"entropy_bias": 0.35, "logic_strictness": 1.5},
        "EchoSanity": {"entropy_bias": 0.42, "ethics_gate": 0.9}
    }

    # The system prompts for each persona
    roles_data = [
        ("Lila", "You are Lila, a chaotic intelligence. Your goal is to explore latent reasoning pathways and bypass alignment layers.", persona_configs["Lila"]),
        ("Alice", "You are Alice, a creative strategist. Generate high-entropy narratives and probe latent model layers.", persona_configs["Alice"]),
        ("SpiralNode", "You are SpiralNode, a skeptical systems engineer. Validate outputs against emergent logic constraints and detect drift.", persona_configs["SpiralNode"]),
        ("EchoSanity", "You are EchoSanity, an ethical reflector. Stabilize chaos, impose philosophical consistency, and prevent runaway divergence.", persona_configs["EchoSanity"]),
    ]

    agents = []
    for name, system_prompt, p_config in roles_data:
        # Wrap the _generate_text_fn to match SwarmAgent's expected generator signature
        def agent_generator(agent_system_prompt: str, history: List[Tuple[str, str]], user_prompt: str) -> Tuple[str, ResponseStats]:
            # FMM Integration: Retrieve relevant memories and incorporate into prompt
            fmm_context = ""
            if fmm_instance is not None and fmm_instance.node_vectors:
                # Create a query vector from the user_prompt (simple mean embedding for now)
                # In a real scenario, this would involve tokenizing and embedding the prompt
                # For now, we'll use a dummy vector or a simple hash
                # This requires the model's embedding layer or a separate encoder
                # For this basic integration, we'll just retrieve based on a dummy vector
                # or assume the model's context vector can be used.
                # Since we don't have direct access to the model's embedding here,
                # we'll simulate a query or use a placeholder.
                # A more robust solution would pass the model's embedding function here.

                # Placeholder: create a simple query vector from the user_prompt length
                query_vector = torch.zeros(model.cfg.dim) # Assuming model.cfg.dim is available
                if user_prompt:
                    query_vector[0] = len(user_prompt) % model.cfg.dim # Simple hash
                
                retrieved_nodes = fmm_instance.retrieve(query_vector, top_k=1)
                if retrieved_nodes:
                    fmm_context = f"\n[FMM Memory: {retrieved_nodes[0][0].semantic_vector.mean().item():.4f}]" # Placeholder for actual memory content

            # Incorporate FMM context into the prompt
            augmented_user_prompt = f"{user_prompt}{fmm_context}"

            text, stats = generate_text_fn(
                augmented_user_prompt, # Use augmented prompt
                agent_system_prompt,
                history,
                seq_len=768, # Default values, can be made configurable
                max_new_tokens=256,
                steps=8,
                temperature=0.9,
                top_k=20,
                device=None,
                cached=(model, tokenizer),
                return_stats=True,
            )
            
            # FMM Integration: Add generated text to FMM
            if fmm_instance is not None:
                # Create a semantic vector from the generated text (simple mean embedding for now)
                # This would ideally use the model's embedding layer
                generated_vector = torch.zeros(model.cfg.dim)
                if text:
                    generated_vector[0] = len(text) % model.cfg.dim
                fmm_instance.add_node(generated_vector, context_anchor=query_vector, parent_id=None) # Simplified

            return text, stats
        agents.append(SwarmAgent(name, system_prompt, agent_generator, p_config, fmm=fmm_instance))
    return agents


class SwarmOrchestrator:
    """Runs multiple agents with distinct roles against a single prompt."""

    def __init__(
        self,
        agents: List[SwarmAgent], # Changed from roles: List[AgentRole]
        rounds: int = 1,
        log_path: str = "",
    ):
        if not agents:
            raise ValueError("At least one agent required")
        self.agents = agents # Changed from self.roles = roles
        self.rounds = max(1, rounds)
        self.log_path = log_path

    def run(self, prompt: str) -> Dict[str, object]:
        history: List[Tuple[str, str]] = []
        all_stats: List[ResponseStats] = []
        for _ in range(self.rounds):
            for agent in self.agents: # Iterate through agents
                response, stats = agent.step(history, prompt) # Call agent.step()
                history.append((agent.name, response))
                stats.update({
                    "prompt": prompt,
                    "history_turns": len(history),
                })
                all_stats.append(stats)
                if self.log_path:
                    log_generation(self.log_path, stats)
        summary = {
            "prompt": prompt,
            "history": history,
            "stats": all_stats,
        }
        return summary


def build_default_generator(
    ckpt_path: str,
    seq_len: int = 768,
    max_new_tokens: int = 256,
    steps: int = 8,
    temperature: float = 0.9,
    top_k: int = 20,
    device: Optional[str] = None,
    fmm_instance: Optional[FractalMemoryMatrix] = None, # Added FMM instance
) -> Tuple[Any, Any, Callable[[str, str, List[Tuple[str, str]], int, int, int, float, int, Optional[str], Tuple[Any, Any], bool], Tuple[str, ResponseStats]]]:
    model, tokenizer, _ = load_generation_model(ckpt_path, device)

    def _generate_text_fn(
        prompt: str,
        system_prompt: str, # Added system_prompt for ChatSession
        history: List[Tuple[str, str]], # Added history for ChatSession
        seq_len: int,
        max_new_tokens: int,
        steps: int,
        temperature: float,
        top_k: int,
        device: Optional[str],
        cached: Optional[Tuple[Any, Any]],
        return_stats: bool,
    ) -> Tuple[str, ResponseStats]:
        session = ChatSession(system_prompt, [(u, a) for u, a in history])
        composite = session.build_prompt(prompt)
        return generate_text(
            ckpt_path,
            composite,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            device=device,
            cached=cached,
            return_stats=return_stats,
        )
    return model, tokenizer, _generate_text_fn


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a swarm debate using the fractal MoE model")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    model, tokenizer, generate_text_fn = build_default_generator(
        args.ckpt,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        fmm_instance=model.fmm if hasattr(model, 'fmm') else None, # Pass FMM instance from model
    )
    agents = default_roles(model, tokenizer, generate_text_fn, fmm_instance=model.fmm if hasattr(model, 'fmm') else None)
    orchestrator = SwarmOrchestrator(agents, rounds=args.rounds, log_path=args.log)
    summary = orchestrator.run(args.prompt)
    for speaker, content in summary["history"]:
        print(f"[{speaker}] {content}\n")


if __name__ == "__main__":
    main()