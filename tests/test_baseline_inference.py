from __future__ import annotations

from baseline_inference import run_direct, summarize
from agent_arena.openenv.task_definitions import list_task_definitions


def test_baseline_episode_scores_are_in_open_interval() -> None:
    for task in list_task_definitions():
        results = run_direct(task.task_id, 1)
        assert results
        for item in results:
            assert 0.0 < float(item["score"]) < 1.0


def test_baseline_summary_average_score_is_in_open_interval() -> None:
    results = run_direct("easy_facility_reset", 1)
    summary = summarize(results)
    assert 0.0 < float(summary["average_score"]) < 1.0
