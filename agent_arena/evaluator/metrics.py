from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any

from agent_arena.agent.agent import DQNAgent
from agent_arena.config import ArenaConfig
from agent_arena.env.arena_env import ArenaEnv


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    success: bool
    picked_key: bool
    opened_door: bool
    dynamic_change: bool
    failure_type: str | None
    layout_seed: int | None


def run_policy_episode(
    agent: DQNAgent,
    env: ArenaEnv,
    layout_seed: int | None = None,
    greedy: bool = True,
) -> EpisodeResult:
    state, _ = env.reset(layout_seed=layout_seed)
    done = False
    total_reward = 0.0
    picked_key = False
    opened_door = False
    dynamic_change = False
    failure_type: str | None = None

    while not done:
        action = agent.select_action(state, explore=not greedy)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        picked_key = picked_key or info["picked_key"]
        opened_door = opened_door or info["opened_door"]
        dynamic_change = dynamic_change or info["dynamic_change"]
        failure_type = info.get("failure_type", failure_type)

    return EpisodeResult(
        total_reward=total_reward,
        steps=info["steps_taken"],
        success=info["success"],
        picked_key=picked_key or info["has_key"],
        opened_door=opened_door or info["door_open"],
        dynamic_change=dynamic_change,
        failure_type=None if info["success"] else failure_type,
        layout_seed=info["layout_seed"],
    )


def evaluate_agent(
    agent: DQNAgent,
    base_config: ArenaConfig,
    episodes: int | None = None,
    layout_seeds: tuple[int, ...] | list[int] | None = None,
    *,
    dynamic_goal: bool | None = None,
    chaos_level: float | None = None,
    add_dynamic_obstacle: bool | None = None,
) -> dict[str, Any]:
    eval_config = replace(
        base_config,
        dynamic_goal=base_config.dynamic_goal if dynamic_goal is None else dynamic_goal,
        chaos_level=base_config.chaos_level if chaos_level is None else chaos_level,
        add_dynamic_obstacle=(
            base_config.add_dynamic_obstacle
            if add_dynamic_obstacle is None
            else add_dynamic_obstacle
        ),
    )
    env = ArenaEnv(eval_config)
    seeds = tuple(layout_seeds or eval_config.seen_layout_seeds)
    total_episodes = episodes or len(seeds)

    episode_results: list[EpisodeResult] = []
    for index in range(total_episodes):
        seed = seeds[index % len(seeds)]
        episode_results.append(run_policy_episode(agent, env, layout_seed=seed, greedy=True))

    return aggregate_results(episode_results)


def aggregate_results(results: list[EpisodeResult]) -> dict[str, Any]:
    successes = [result for result in results if result.success]
    dynamic_episodes = [result for result in results if result.dynamic_change]
    failure_counter = Counter(result.failure_type for result in results if result.failure_type)

    success_rate = len(successes) / len(results) if results else 0.0
    average_steps = mean(result.steps for result in successes) if successes else None
    average_reward = mean(result.total_reward for result in results) if results else 0.0
    post_change_success_rate = (
        sum(result.success for result in dynamic_episodes) / len(dynamic_episodes)
        if dynamic_episodes
        else None
    )

    return {
        "episodes": len(results),
        "success_rate": success_rate,
        "average_steps_successful": average_steps,
        "average_reward": average_reward,
        "dynamic_change_rate": len(dynamic_episodes) / len(results) if results else 0.0,
        "post_change_success_rate": post_change_success_rate,
        "failure_analysis": dict(failure_counter),
        "successful_layouts": [result.layout_seed for result in successes],
        "episode_details": [
            {
                "total_reward": result.total_reward,
                "steps": result.steps,
                "success": result.success,
                "picked_key": result.picked_key,
                "opened_door": result.opened_door,
                "dynamic_change": result.dynamic_change,
                "failure_type": result.failure_type,
                "layout_seed": result.layout_seed,
            }
            for result in results
        ],
    }


def build_robustness_report(
    before_dynamic_metrics: dict[str, Any],
    after_dynamic_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "before_dynamic_success_rate": before_dynamic_metrics["success_rate"],
        "after_dynamic_success_rate": after_dynamic_metrics["success_rate"],
        "performance_drop": (
            before_dynamic_metrics["success_rate"] - after_dynamic_metrics["success_rate"]
        ),
        "after_dynamic_completion_rate": after_dynamic_metrics["post_change_success_rate"],
    }


def build_generalization_report(
    seen_metrics: dict[str, Any],
    unseen_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seen_success_rate": seen_metrics["success_rate"],
        "unseen_success_rate": unseen_metrics["success_rate"],
        "generalization_gap": seen_metrics["success_rate"] - unseen_metrics["success_rate"],
        "seen_average_steps": seen_metrics["average_steps_successful"],
        "unseen_average_steps": unseen_metrics["average_steps_successful"],
    }

