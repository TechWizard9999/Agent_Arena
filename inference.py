from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any

from openai import OpenAI

from agent_arena.openenv.task_definitions import get_task_definition, list_task_definitions
from baseline_inference import _action_from_id, choose_baseline_action
from client import AgentArenaEnv
from server.agent_arena_environment import AgentArenaEnvironment


DEFAULT_ENV_BASE_URL = "http://127.0.0.1:8000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submission inference runner with structured logs for Agent Arena.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("ENV_BASE_URL"),
        help="Optional running environment URL. When omitted, inference uses the local direct environment.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=4,
        help="How many deterministic episodes to run per task.",
    )
    return parser.parse_args()


def build_openai_client() -> OpenAI | None:
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_base_url or not api_key or not model_name:
        return None

    return OpenAI(base_url=api_base_url, api_key=api_key)


def log_event(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def run_direct(task_id: str, episodes: int) -> list[dict[str, Any]]:
    env = AgentArenaEnvironment()
    task = get_task_definition(task_id)
    results: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        layout_seed = task.layout_seeds[episode_index % len(task.layout_seeds)]
        observation = env.reset(task_id=task_id, layout_seed=layout_seed)
        log_event(
            "[START]",
            {
                "task_id": task_id,
                "episode": episode_index + 1,
                "layout_seed": layout_seed,
                "mode": "direct",
                "success_threshold": task.success_threshold,
            },
        )

        done = observation.done
        step_count = 0
        final_observation = observation

        while not done:
            action_id = choose_baseline_action(env._env)
            action = _action_from_id(action_id)
            final_observation = env.step(action)
            step_count += 1

            log_event(
                "[STEP]",
                {
                    "task_id": task_id,
                    "episode": episode_index + 1,
                    "step": step_count,
                    "action": action.action.value,
                    "reward": round(float(final_observation.reward or 0.0), 6),
                    "score": round(float(final_observation.score), 6),
                    "status": final_observation.status,
                    "done": final_observation.done,
                },
            )
            done = final_observation.done

        episode_result = {
            "task_id": task_id,
            "layout_seed": layout_seed,
            "score": float(final_observation.score),
            "passed": bool(final_observation.done and final_observation.score >= task.success_threshold),
            "reward": float(final_observation.reward or 0.0),
            "steps": step_count,
            "failure_type": final_observation.metadata.get("failure_type"),
        }
        log_event("[END]", episode_result)
        results.append(episode_result)

    return results


def run_remote(base_url: str, task_id: str, episodes: int) -> list[dict[str, Any]]:
    task = get_task_definition(task_id)
    results: list[dict[str, Any]] = []

    with AgentArenaEnv(base_url=base_url).sync() as client:
        for episode_index in range(episodes):
            layout_seed = task.layout_seeds[episode_index % len(task.layout_seeds)]
            result = client.reset(task_id=task_id, layout_seed=layout_seed)
            log_event(
                "[START]",
                {
                    "task_id": task_id,
                    "episode": episode_index + 1,
                    "layout_seed": layout_seed,
                    "mode": "remote",
                    "base_url": base_url,
                    "success_threshold": task.success_threshold,
                },
            )

            done = result.done
            step_count = 0
            final_observation = result.observation

            while not done:
                helper_env = AgentArenaEnvironment()
                helper_env.reset(task_id=task_id, layout_seed=layout_seed)
                while helper_env.state.step_count < client.state().step_count:
                    helper_action = _action_from_id(choose_baseline_action(helper_env._env))
                    helper_env.step(helper_action)

                action_id = choose_baseline_action(helper_env._env)
                action = _action_from_id(action_id)
                result = client.step(action)
                final_observation = result.observation
                step_count += 1

                log_event(
                    "[STEP]",
                    {
                        "task_id": task_id,
                        "episode": episode_index + 1,
                        "step": step_count,
                        "action": action.action.value,
                        "reward": round(float(result.reward or 0.0), 6),
                        "score": round(float(final_observation.score), 6),
                        "status": final_observation.status,
                        "done": result.done,
                    },
                )
                done = result.done

            episode_result = {
                "task_id": task_id,
                "layout_seed": layout_seed,
                "score": float(final_observation.score),
                "passed": bool(final_observation.score >= task.success_threshold),
                "reward": float(result.reward or 0.0),
                "steps": step_count,
                "failure_type": final_observation.metadata.get("failure_type"),
            }
            log_event("[END]", episode_result)
            results.append(episode_result)

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "episodes": 0,
            "average_score": 0.0,
            "pass_rate": 0.0,
            "average_steps": 0.0,
            "details": [],
        }

    return {
        "episodes": len(results),
        "average_score": mean(item["score"] for item in results),
        "pass_rate": mean(1.0 if item["passed"] else 0.0 for item in results),
        "average_steps": mean(item["steps"] for item in results),
        "details": results,
    }


def main() -> None:
    args = parse_args()
    client = build_openai_client()

    log_event(
        "[START]",
        {
            "script": "inference.py",
            "env_mode": "remote" if args.base_url else "direct",
            "base_url": args.base_url or DEFAULT_ENV_BASE_URL,
            "episodes_per_task": args.episodes_per_task,
            "openai_client_configured": client is not None,
        },
    )

    payload: dict[str, Any] = {"tasks": {}}

    for task in list_task_definitions():
        if args.base_url:
            results = run_remote(args.base_url, task.task_id, args.episodes_per_task)
        else:
            results = run_direct(task.task_id, args.episodes_per_task)
        payload["tasks"][task.task_id] = summarize(results)

    log_event("[END]", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
