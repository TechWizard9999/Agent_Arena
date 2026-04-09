from __future__ import annotations

from models import AgentArenaAction, AgentArenaActionType
from server.agent_arena_environment import AgentArenaEnvironment


def test_reset_score_is_in_open_interval() -> None:
    env = AgentArenaEnvironment()
    observation = env.reset(task_id="easy_facility_reset", layout_seed=11)
    assert 0.0 < observation.score < 1.0
    assert 0.0 < env.state.score < 1.0


def test_step_scores_remain_in_open_interval() -> None:
    env = AgentArenaEnvironment()
    env.reset(task_id="easy_facility_reset", layout_seed=11)

    for _ in range(32):
        observation = env.step(AgentArenaAction(action=AgentArenaActionType.UP))
        assert 0.0 < observation.score < 1.0
        assert 0.0 < env.state.score < 1.0
        if observation.done:
            break


def test_invalid_difficulty_scale_rejects_nan() -> None:
    env = AgentArenaEnvironment()
    try:
        env.reset(difficulty_scale=float("nan"))
    except ValueError as exc:
        assert "difficulty_scale" in str(exc)
    else:
        raise AssertionError("Expected difficulty_scale=float('nan') to be rejected.")
