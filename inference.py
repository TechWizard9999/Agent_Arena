from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any

from openai import OpenAI

from agent_arena.openenv.task_definitions import get_task_definition, list_task_definitions
from baseline_inference import _action_from_id, choose_baseline_action
from client import AgentArenaEnv
from server.agent_arena_environment import AgentArenaEnvironment


DEFAULT_ENV_BASE_URL = "http://127.0.0.1:7860"
ACTION_ORDER = ("pick_badge", "open_gate", "up", "down", "left", "right")


@dataclass(frozen=True)
class LLMRuntime:
    client: OpenAI
    model_name: str


SYSTEM_PROMPT = """
You are controlling a maintenance rover in a facility operations benchmark.
Goal order matters: collect the badge, open the gate, then reach the checkpoint.
Return exactly one legal action token and nothing else.
Allowed actions are: up, down, left, right, pick_badge, open_gate.
If unsure, return the provided heuristic_suggestion.
""".strip()


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


def build_openai_client() -> LLMRuntime | None:
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_base_url or not api_key or not model_name:
        return None

    return LLMRuntime(
        client=OpenAI(base_url=api_base_url, api_key=api_key),
        model_name=model_name,
    )


def log_event(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def _call_llm(runtime: LLMRuntime, *, system_prompt: str, user_prompt: str) -> str:
    response = runtime.client.chat.completions.create(
        model=runtime.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=12,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _verify_llm_proxy(runtime: LLMRuntime) -> None:
    _call_llm(
        runtime,
        system_prompt="Reply with the single token OK.",
        user_prompt="Health-check the proxy path. Return only OK.",
    )


def _extract_action(reply_text: str, legal_actions: list[str], fallback: str) -> str:
    normalized = reply_text.strip().lower()
    for action_name in ACTION_ORDER:
        if action_name in legal_actions and action_name in normalized:
            return action_name
    return fallback


def _policy_action_name(
    *,
    runtime: LLMRuntime | None,
    observation: Any,
    heuristic_action_name: str,
) -> tuple[str, str]:
    if runtime is None:
        return heuristic_action_name, "heuristic"

    payload = {
        "task_id": observation.task_id,
        "task_prompt": observation.task_prompt,
        "status": observation.status,
        "legal_actions": observation.legal_actions,
        "agent_position": observation.agent_position,
        "badge_position": observation.badge_position,
        "gate_position": observation.gate_position,
        "checkpoint_position": observation.checkpoint_position,
        "has_badge": observation.has_badge,
        "gate_open": observation.gate_open,
        "dynamic_event_triggered": observation.dynamic_event_triggered,
        "steps_remaining": observation.steps_remaining,
        "event_log": observation.event_log[-3:],
        "heuristic_suggestion": heuristic_action_name,
    }
    reply_text = _call_llm(
        runtime,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, separators=(",", ":")),
    )
    return _extract_action(reply_text, observation.legal_actions, heuristic_action_name), "llm"


def run_direct(task_id: str, episodes: int, runtime: LLMRuntime | None) -> list[dict[str, Any]]:
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
            heuristic_action_id = choose_baseline_action(env._env)
            heuristic_action = _action_from_id(heuristic_action_id)
            action_name, policy_source = _policy_action_name(
                runtime=runtime,
                observation=final_observation,
                heuristic_action_name=heuristic_action.action.value,
            )
            action = _action_from_id(
                heuristic_action_id
                if action_name == heuristic_action.action.value
                else next(
                    action_id
                    for action_id in range(6)
                    if _action_from_id(action_id).action.value == action_name
                )
            )
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
                    "policy_source": policy_source,
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


def run_remote(base_url: str, task_id: str, episodes: int, runtime: LLMRuntime | None) -> list[dict[str, Any]]:
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

                heuristic_action_id = choose_baseline_action(helper_env._env)
                heuristic_action = _action_from_id(heuristic_action_id)
                action_name, policy_source = _policy_action_name(
                    runtime=runtime,
                    observation=final_observation,
                    heuristic_action_name=heuristic_action.action.value,
                )
                action = _action_from_id(
                    heuristic_action_id
                    if action_name == heuristic_action.action.value
                    else next(
                        action_id
                        for action_id in range(6)
                        if _action_from_id(action_id).action.value == action_name
                    )
                )
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
                    "policy_source": policy_source,
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
    runtime = build_openai_client()

    if runtime is not None:
        _verify_llm_proxy(runtime)

    log_event(
        "[START]",
        {
            "script": "inference.py",
            "env_mode": "remote" if args.base_url else "direct",
            "base_url": args.base_url or DEFAULT_ENV_BASE_URL,
            "episodes_per_task": args.episodes_per_task,
            "openai_client_configured": runtime is not None,
            "llm_policy_enabled": runtime is not None,
        },
    )

    payload: dict[str, Any] = {"tasks": {}}

    for task in list_task_definitions():
        if args.base_url:
            results = run_remote(args.base_url, task.task_id, args.episodes_per_task, runtime)
        else:
            results = run_direct(task.task_id, args.episodes_per_task, runtime)
        payload["tasks"][task.task_id] = summarize(results)

    log_event("[END]", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
