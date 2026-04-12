from __future__ import annotations

from dataclasses import dataclass

from agent_arena.openenv.task_definitions import TaskDefinition

OPEN_SCORE_EPSILON = 0.01


@dataclass(frozen=True, eq=False)
class GradeResult:
    """Normalized task score returned by the OpenEnv grader.

    Supports numeric operations so external validators can use the result
    directly as a float (e.g. ``0 < grade_episode(...) < 1``).
    """

    score: float
    passed: bool
    breakdown: dict[str, float]

    def __float__(self) -> float:
        return self.score

    def __int__(self) -> int:
        return int(self.score)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GradeResult):
            return self.score == other.score
        if isinstance(other, (int, float)):
            return self.score == float(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.score, self.passed, tuple(sorted(self.breakdown.items()))))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float, GradeResult)):
            return self.score < float(other)
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, (int, float, GradeResult)):
            return self.score <= float(other)
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float, GradeResult)):
            return self.score > float(other)
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (int, float, GradeResult)):
            return self.score >= float(other)
        return NotImplemented

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def clamp_open_score(value: float) -> float:
    """Clamp a score into the strict open interval required by the validator."""
    return min(1.0 - OPEN_SCORE_EPSILON, max(OPEN_SCORE_EPSILON, value))


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
    efficiency = (
        0.10 * max(0.0, 1.0 - (steps_taken / max(1, max_steps)))
        if success
        else 0.0
    )

    raw_total = milestone_badge + milestone_gate + milestone_checkpoint + efficiency
    score = clamp_open_score(raw_total)
    # Keep the breakdown sum identical to the returned score so downstream
    # validators that reconstruct scores from components stay in sync.
    efficiency += score - raw_total

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
