from fractal_neurons.orchestrator import SwarmOrchestrator, AgentRole


def dummy_generator(system_prompt, history, prompt):
    reply = f"{system_prompt[:10]}|{prompt[:5]}|{len(history)}"
    stats = {"system": system_prompt, "history_len": len(history)}
    return reply, stats


def test_swarm_orchestrator_runs_rounds():
    roles = [AgentRole("Planner", "plan"), AgentRole("Critic", "critic")]
    orch = SwarmOrchestrator(roles, dummy_generator, rounds=2)
    summary = orch.run("hello world")
    assert len(summary["history"]) == 4
    assert summary["stats"][0]["history_len"] == 0
