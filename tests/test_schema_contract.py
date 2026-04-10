from __future__ import annotations

from models import AgentArenaObservation, AgentArenaState


def test_observation_score_schema_uses_exclusive_bounds() -> None:
    score_schema = AgentArenaObservation.model_json_schema()["properties"]["score"]

    assert score_schema["exclusiveMinimum"] == 0.0
    assert score_schema["exclusiveMaximum"] == 1.0
    assert "default" not in score_schema


def test_state_score_schema_uses_exclusive_bounds() -> None:
    score_schema = AgentArenaState.model_json_schema()["properties"]["score"]

    assert score_schema["exclusiveMinimum"] == 0.0
    assert score_schema["exclusiveMaximum"] == 1.0
    assert "default" not in score_schema
