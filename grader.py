from __future__ import annotations

from dataclasses import dataclass

from agent_arena.openenv.task_definitions import TaskDefinition


@dataclass(frozen=True)
class GradeResult:
    """Normalized task score returned by the OpenEnv grader."""

    score: float
    passed: bool
    breakdown: dict[str, float]


def grade_episode(
    *,
    has_badge: bool,
    gate_open: bool,
    success: bool,
    steps_taken: int,
    max_steps: int,
) -> GradeResult:
    milestone_badge = 0.30 if has_badge else 0.0
    milestone_gate = 0.30 if gate_open else 0.0
    milestone_checkpoint = 0.30 if success else 0.0
    efficiency = 0.10 * max(0.0, 1.0 - (steps_taken / max_steps)) if success else 0.0

    raw_score = milestone_badge + milestone_gate + milestone_checkpoint + efficiency
    
    # CLAMP THE SCORE TO (0.01, 0.99)
    # This prevents the score from being exactly 0.0 or exactly 1.0
    score = max(0.01, min(0.99, raw_score))
    
    return GradeResult(
        score=score,
        passed=success,
        breakdown={
            "badge": milestone_badge,
            "gate": milestone_gate,
            "checkpoint": milestone_checkpoint,
            "efficiency": efficiency,
        },
    )


def normalized_step_reward(
    previous_score: float,
    current_score: float,
    invalid_action: bool,
) -> float:
    if invalid_action:
        return 0.0
    return max(0.0, min(1.0, current_score - previous_score))


def task_summary(task: TaskDefinition) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "difficulty": task.difficulty,
        "prompt": task.prompt,
        "dynamic_goal": task.dynamic_goal,
        "chaos_level": task.chaos_level,
        "add_dynamic_obstacle": task.add_dynamic_obstacle,
        "dynamic_change_step": task.dynamic_change_step,
        "max_steps": task.max_steps,
        "layout_seeds": list(task.layout_seeds),
        "success_criteria": task.success_criteria,
        "evaluation_dimensions": list(task.evaluation_dimensions),
        "expected_baseline_score_range": list(task.expected_baseline_score_range),
        "success_threshold": task.success_threshold,
    }