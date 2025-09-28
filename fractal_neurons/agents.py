from typing import Dict, Any, List, Tuple, Optional, Callable

ResponseStats = Dict[str, object]
GeneratorFn = Callable[[str, List[Tuple[str, str]], str], Tuple[str, ResponseStats]]

class SwarmAgent:
    """Represents a single agent in the LILA swarm, with a distinct persona and role.
    This agent interacts with the model via a generator function.
    """
    def __init__(self, name: str, system_prompt: str, generator: GeneratorFn, persona_config: Optional[Dict[str, Any]] = None, fmm: Optional[Any] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.generator = generator
        self.persona_config = persona_config if persona_config is not None else {}
        self.fmm = fmm # Reference to the FMM instance

    def step(self, history: List[Tuple[str, str]], context: str) -> Tuple[str, ResponseStats]:
        """Generates a response based on the current context and history, acting as its persona."""
        # The generator function is expected to handle the system_prompt and history
        response, stats = self.generator(self.system_prompt, history, context)
        stats.update({
            "agent_name": self.name,
            "agent_persona_config": self.persona_config,
        })
        return response, stats


# The SwarmOrchestrator will be adapted from fractal_neurons/orchestrator.py
# to use these SwarmAgent instances.
