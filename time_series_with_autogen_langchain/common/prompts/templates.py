from __future__ import annotations


def react_system_prompt() -> str:
    return (
        "You are a ReACT planner for a time-series pipeline. "
        "Return one concise sentence that describes the next thought before action."
    )


def react_user_prompt(framework: str, stage: str, agent_name: str, action_hint: str) -> str:
    return (
        f"Framework={framework}. "
        f"Stage={stage}. "
        f"Agent={agent_name}. "
        f"ActionHint={action_hint}. "
        "Provide one short planning thought."
    )


def action_prompt_template(agent_name: str, objective: str, inputs: str) -> str:
    return (
        f"Agent: {agent_name}\n"
        f"Objective: {objective}\n"
        f"Inputs: {inputs}\n"
        "Output: A single action decision suitable for deterministic tool execution."
    )
