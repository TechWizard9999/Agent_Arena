from __future__ import annotations

import argparse
import json
from statistics import mean

from agent_arena.env.arena_env import ArenaEnv
from agent_arena.env.objects import Position
from agent_arena.openenv.grader import clamp_open_score
from agent_arena.openenv.task_definitions import (
    build_task_config,
    get_task_definition,
    list_task_definitions,
)
from agent_arena.trainer.train import choose_expert_action
from client import AgentArenaEnv
from models import AgentArenaAction
from server.agent_arena_environment import ACTION_TO_ID, AgentArenaEnvironment


def safe_score(value: float) -> float:
    return clamp_open_score(float(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible baseline for Agent Arena.")
    parser.add_argument("--base-url", type=str, default=None, help="Optional remote OpenEnv URL.")
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=4,
        help="Episodes to run for each task.",
    )
    return parser.parse_args()


def _action_from_id(action_id: int) -> AgentArenaAction:
    for action_type, mapped_id in ACTION_TO_ID.items():
        if mapped_id == action_id:
            return AgentArenaAction(action=action_type)
    raise ValueError(f"Unknown action id: {action_id}")


def _is_move_valid(env: ArenaEnv, candidate: Position) -> bool:
    if env.layout is None:
        return False
    if not (0 <= candidate.x < env.grid_size and 0 <= candidate.y < env.grid_size):
        return False
    if candidate in env.layout.structural_walls or candidate in env.layout.obstacles:
        return False
    if candidate == env.layout.door and not env.door_open:
        return False
    return True


def choose_baseline_action(env: ArenaEnv) -> int:
    try:
        return choose_expert_action(env)
    except RuntimeError:
        if env.layout is None or env.agent_pos is None:
            return 0

        if not env.has_key and env.agent_pos == env.layout.key:
            return 4
        if env.has_key and not env.door_open and env.agent_pos.manhattan(env.layout.door) == 1:
            return 5

        if not env.has_key:
            target = env.layout.key
        elif not env.door_open:
            target = env.layout.door
        else:
            target = env.layout.goal

        best_action = 0
        best_distance = float("inf")
        for action_id, (dx, dy) in env.ACTIONS.items():
            candidate = env.agent_pos.moved(dx, dy)
            if not _is_move_valid(env, candidate):
                continue
            distance = candidate.manhattan(target)
            if distance < best_distance:
                best_distance = distance
                best_action = action_id
        return best_action


def run_direct(task_id: str, episodes: int) -> list[dict[str, object]]:
    env = AgentArenaEnvironment()
    results: list[dict[str, object]] = []
    task = get_task_definition(task_id)

    for episode in range(episodes):
        layout_seed = task.layout_seeds[episode % len(task.layout_seeds)]
        env.reset(task_id=task_id, layout_seed=layout_seed)
        core_env = env._env
        done = False
        final_observation = None

        while not done:
            action_id = choose_baseline_action(core_env)
            action = _action_from_id(action_id)
            final_observation = env.step(action)
            done = final_observation.done

        if final_observation is None:
            raise RuntimeError("Baseline rollout did not produce any observation.")

        breakdown = final_observation.metadata.get("grade_breakdown", {})
        results.append(
            {
                "task_id": task_id,
                "layout_seed": layout_seed,
                "score": safe_score(final_observation.score),
                "passed": final_observation.done and final_observation.score >= task.success_threshold,
                "reward": final_observation.reward,
                "breakdown": breakdown,
            }
        )

    return results


def run_remote(base_url: str, task_id: str, episodes: int) -> list[dict[str, object]]:
    task = next(task for task in list_task_definitions() if task.task_id == task_id)
    results: list[dict[str, object]] = []

    with AgentArenaEnv(base_url=base_url).sync() as client:
        for episode in range(episodes):
            layout_seed = task.layout_seeds[episode % len(task.layout_seeds)]
            result = client.reset(task_id=task_id, layout_seed=layout_seed)
            done = result.done
            final_observation = result.observation

            while not done:
                config = build_task_config(task_id)
                direct_env = ArenaEnv(config)
                direct_env.reset(layout_seed=layout_seed)
                # Replay remote progress into a local helper env to reuse the deterministic expert.
                # For reproducibility we continue using the local environment state only as an oracle.
                # The expert policy is deterministic for a given layout seed.
                while direct_env.steps_taken < client.state().step_count:
                    action_id = choose_baseline_action(direct_env)
                    direct_env.step(action_id)

                action_id = choose_baseline_action(direct_env)
                result = client.step(_action_from_id(action_id))
                final_observation = result.observation
                done = result.done

            results.append(
                {
                    "task_id": task_id,
                    "layout_seed": layout_seed,
                    "score": safe_score(final_observation.score),
                    "passed": final_observation.score >= task.success_threshold,
                    "reward": final_observation.reward if final_observation.reward is not None else final_observation.score,
                    "breakdown": final_observation.metadata.get("grade_breakdown", {}),
                }
            )

    return results


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    task_id = str(results[0]["task_id"]) if results else "easy_facility_reset"
    task = get_task_definition(task_id)
    average_score = (
        safe_score(mean(float(item["score"]) for item in results))
        if results
        else safe_score(0.0)
    )
    low, high = task.expected_baseline_score_range
    return {
        "episodes": len(results),
        "average_score": average_score,
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results) if results else 0.0,
        "expected_baseline_score_range": list(task.expected_baseline_score_range),
        "within_expected_range": low <= average_score <= high,
        "success_threshold": task.success_threshold,
        "episodes_detail": results,
    }


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"tasks": {}}

    for task in list_task_definitions():
        if args.base_url:
            results = run_remote(args.base_url, task.task_id, args.episodes_per_task)
        else:
            results = run_direct(task.task_id, args.episodes_per_task)
        payload["tasks"][task.task_id] = summarize(results)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
