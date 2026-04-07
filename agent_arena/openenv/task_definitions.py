from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    success_criteria: str
    evaluation_dimensions: tuple[str, ...]
    expected_baseline_score_range: tuple[float, float]
    success_threshold: float = 0.9


ACTION_CONTEXT: Mapping[str, str] = {
    "up": "Move the maintenance rover one cell north through the facility.",
    "down": "Move the maintenance rover one cell south through the facility.",
    "left": "Move the maintenance rover one cell west through the facility.",
    "right": "Move the maintenance rover one cell east through the facility.",
    "pick_badge": "Collect the access badge when the rover is standing on it.",
    "open_gate": "Unlock the safety gate when the rover is adjacent and already has the badge.",
}


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
        success_criteria=(
            "Pass by collecting the access badge, opening the safety gate, and reaching the "
            "checkpoint within the step budget."
        ),
        evaluation_dimensions=("sequencing", "goal completion", "efficiency"),
        expected_baseline_score_range=(0.94, 0.99),
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
        success_criteria=(
            "Pass by completing the badge -> gate -> checkpoint sequence even if control reroutes "
            "the checkpoint after the episode begins."
        ),
        evaluation_dimensions=("planning", "adaptation", "reroute recovery", "efficiency"),
        expected_baseline_score_range=(0.60, 0.92),
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
        success_criteria=(
            "Pass by recovering from both checkpoint reroutes and aisle disruptions while still "
            "completing the full facility mission inside the tighter step budget."
        ),
        evaluation_dimensions=("long-horizon planning", "adaptation", "robustness", "efficiency"),
        expected_baseline_score_range=(0.68, 0.82),
    ),
}

DIFFICULTY_BUCKET_TO_TASK_ID: Mapping[str, str] = {
    "easy": "easy_facility_reset",
    "medium": "medium_reroute_response",
    "hard": "hard_disruption_recovery",
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


def build_curriculum_config(difficulty: float) -> ArenaConfig:
    difficulty = max(0.0, min(1.0, difficulty))

    chaos_level = 0.60 * difficulty
    dynamic_change_step = max(6, round(12 - (6 * difficulty)))
    max_steps = max(28, round(38 - (10 * difficulty)))

    return ArenaConfig(
        grid_size=5,
        dynamic_goal=difficulty >= 0.2,
        dynamic_change_step=dynamic_change_step,
        chaos_level=chaos_level,
        add_dynamic_obstacle=difficulty >= 0.45,
        max_steps=max_steps,
        seed=7,
    )


def infer_difficulty_bucket(difficulty: float) -> str:
    difficulty = max(0.0, min(1.0, difficulty))
    if difficulty < 0.34:
        return "easy"
    if difficulty < 0.67:
        return "medium"
    return "hard"


def curriculum_summary(difficulty: float) -> dict[str, object]:
    config = build_curriculum_config(difficulty)
    return {
        "difficulty_scale": round(max(0.0, min(1.0, difficulty)), 3),
        "difficulty_bucket": infer_difficulty_bucket(difficulty),
        "task_id": DIFFICULTY_BUCKET_TO_TASK_ID[infer_difficulty_bucket(difficulty)],
        "dynamic_goal": config.dynamic_goal,
        "chaos_level": config.chaos_level,
        "add_dynamic_obstacle": config.add_dynamic_obstacle,
        "dynamic_change_step": config.dynamic_change_step,
        "max_steps": config.max_steps,
    }


def list_task_definitions() -> list[TaskDefinition]:
    return [TASK_DEFINITIONS[key] for key in sorted(TASK_DEFINITIONS)]
