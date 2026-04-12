from __future__ import annotations

import math

import pytest

from agent_arena.openenv.grader import OPEN_SCORE_EPSILON, grade_episode


def test_total_failure_score_is_strictly_above_zero() -> None:
    result = grade_episode(
        has_badge=False,
        gate_open=False,
        success=False,
        steps_taken=32,
        max_steps=32,
    )
    assert result.score == OPEN_SCORE_EPSILON
    assert 0.0 < result.score < 1.0


def test_perfect_score_is_strictly_below_one() -> None:
    result = grade_episode(
        has_badge=True,
        gate_open=True,
        success=True,
        steps_taken=0,
        max_steps=32,
    )
    assert result.score == 1.0 - OPEN_SCORE_EPSILON
    assert 0.0 < result.score < 1.0


def test_total_failure_breakdown_sums_to_clamped_score() -> None:
    result = grade_episode(
        has_badge=False,
        gate_open=False,
        success=False,
        steps_taken=32,
        max_steps=32,
    )
    assert math.isclose(sum(result.breakdown.values()), result.score, rel_tol=0.0, abs_tol=1e-12)


def test_perfect_breakdown_sums_to_clamped_score() -> None:
    result = grade_episode(
        has_badge=True,
        gate_open=True,
        success=True,
        steps_taken=0,
        max_steps=32,
    )
    assert math.isclose(sum(result.breakdown.values()), result.score, rel_tol=0.0, abs_tol=1e-12)


def test_zero_max_steps_does_not_crash() -> None:
    result = grade_episode(
        has_badge=True,
        gate_open=True,
        success=True,
        steps_taken=1,
        max_steps=0,
    )
    assert 0.0 < result.score < 1.0


@pytest.mark.parametrize("steps,max_steps", [(0, 1), (1, 1), (10, 32), (100, 100)])
def test_score_always_stays_in_open_interval(steps: int, max_steps: int) -> None:
    for has_badge in (False, True):
        for gate_open in (False, True):
            for success in (False, True):
                result = grade_episode(
                    has_badge=has_badge,
                    gate_open=gate_open,
                    success=success,
                    steps_taken=steps,
                    max_steps=max_steps,
                )
                assert 0.0 < result.score < 1.0
                assert math.isclose(
                    sum(result.breakdown.values()),
                    result.score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
