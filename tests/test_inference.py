from __future__ import annotations

from inference import format_score, render_submission_payload, safe_score


def test_safe_score_clamps_non_finite_values_into_open_interval() -> None:
    assert 0.0 < safe_score(float("nan")) < 1.0
    assert 0.0 < safe_score(float("inf")) < 1.0


def test_format_score_uses_fixed_decimal_notation() -> None:
    assert format_score(0.0) == "0.000001"
    assert format_score(1.0) == "0.999999"


def test_render_submission_payload_uses_fixed_decimal_scores() -> None:
    payload = {
        "tasks": {
            "easy_facility_reset": {"score": 0.0},
            "medium_reroute_response": {"score": 1.0},
        }
    }

    rendered = render_submission_payload(payload)

    assert '"easy_facility_reset": {"score": 0.000001}' in rendered
    assert '"medium_reroute_response": {"score": 0.999999}' in rendered
    assert "1e-06" not in rendered
