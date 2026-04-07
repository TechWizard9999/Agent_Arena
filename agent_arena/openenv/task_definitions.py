from __future__ import annotations

from dataclasses import dataclass

from agent_arena.config import ArenaConfig


@dataclass(frozen=True)
class TaskDefinition:
    """OpenEnv-facing task definition with normalized scoring expectations."""

    task_id: str
    title: str
    difficulty: str
    prompt: str
    dynamic_goal: bool
    chaos_level: float
    add_dynamic_obstacle: bool
    dynamic_change_step: int
    max_steps: int
    layout_seeds: tuple[int, ...]


TASK_DEFINITIONS: dict[str, TaskDefinition] = {
    "easy_facility_reset": TaskDefinition(
        task_id="easy_facility_reset",
        title="Easy Facility Reset",
        difficulty="easy",
        prompt=(
            "You are an autonomous maintenance rover in a secure industrial facility. "
            "Collect the access badge, unlock the safety gate, and reach the control "
            "checkpoint before the shift timer expires."
        ),
        dynamic_goal=False,
        chaos_level=0.0,
        add_dynamic_obstacle=False,
        dynamic_change_step=99,
        max_steps=32,
        layout_seeds=(11, 17, 23, 31),
    ),
    "medium_reroute_response": TaskDefinition(
        task_id="medium_reroute_response",
        title="Medium Reroute Response",
        difficulty="medium",
        prompt=(
            "A facility alert may reroute the inspection target during execution. "
            "Collect the access badge, unlock the safety gate, and adapt to the "
            "new checkpoint if control redirects your assignment."
        ),
        dynamic_goal=True,
        chaos_level=0.25,
        add_dynamic_obstacle=False,
        dynamic_change_step=8,
        max_steps=34,
        layout_seeds=(43, 59, 67, 79),
    ),
    "hard_disruption_recovery": TaskDefinition(
        task_id="hard_disruption_recovery",
        title="Hard Disruption Recovery",
        difficulty="hard",
        prompt=(
            "The facility is unstable: aisle blockages can appear and the repair target "
            "may move after an alarm. Retrieve the access badge, unlock the gate, and "
            "recover the service route fast enough to finish the mission."
        ),
        dynamic_goal=True,
        chaos_level=0.55,
        add_dynamic_obstacle=True,
        dynamic_change_step=7,
        max_steps=30,
        layout_seeds=(101, 113, 127, 149),
    ),
}


def get_task_definition(task_id: str) -> TaskDefinition:
    try:
        return TASK_DEFINITIONS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_DEFINITIONS))
        raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {available}") from exc


def build_task_config(task_id: str) -> ArenaConfig:
    task = get_task_definition(task_id)
    return ArenaConfig(
        grid_size=5,
        dynamic_goal=task.dynamic_goal,
        dynamic_change_step=task.dynamic_change_step,
        chaos_level=task.chaos_level,
        add_dynamic_obstacle=task.add_dynamic_obstacle,
        max_steps=task.max_steps,
        seed=7,
    )


def list_task_definitions() -> list[TaskDefinition]:
    return [TASK_DEFINITIONS[key] for key in sorted(TASK_DEFINITIONS)]

