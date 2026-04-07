from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from agent_arena.agent.agent import DQNAgent
from agent_arena.env.objects import Position
from agent_arena.config import ArenaConfig
from agent_arena.env.arena_env import ArenaEnv
from agent_arena.evaluator.metrics import (
    build_generalization_report,
    build_robustness_report,
    evaluate_agent,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_agent(
    config: ArenaConfig,
    run_name: str,
    layout_seeds: tuple[int, ...] | None = None,
) -> tuple[DQNAgent, dict[str, Any]]:
    set_global_seed(config.seed)
    env = ArenaEnv(config)
    agent = DQNAgent(env.state_size, env.action_size, config)
    seeds = layout_seeds or config.seen_layout_seeds

    reward_history: list[float] = []
    loss_history: list[float | None] = []
    success_history: list[int] = []
    failure_history: list[str] = []
    bootstrap_summary = bootstrap_agent(agent, config, seeds) if config.use_expert_bootstrap else {}

    for episode in range(config.train_episodes):
        layout_seed = seeds[episode % len(seeds)]
        state, _ = env.reset(layout_seed=layout_seed)
        done = False
        total_reward = 0.0
        episode_losses: list[float] = []
        final_info: dict[str, Any] = {}

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            final_info = info

        agent.decay_epsilon()
        if (episode + 1) % config.target_update_frequency == 0:
            agent.update_target_network()

        reward_history.append(total_reward)
        loss_history.append(float(np.mean(episode_losses)) if episode_losses else None)
        success_history.append(int(final_info.get("success", False)))
        if final_info.get("failure_type"):
            failure_history.append(final_info["failure_type"])

        if (episode + 1) % config.log_interval == 0 or episode == 0:
            recent_rewards = reward_history[-config.log_interval :]
            recent_successes = success_history[-config.log_interval :]
            print(
                f"[{run_name}] Episode {episode + 1}/{config.train_episodes} "
                f"| avg_reward={np.mean(recent_rewards):.2f} "
                f"| success_rate={np.mean(recent_successes):.2f} "
                f"| epsilon={agent.epsilon:.3f}"
            )

    training_summary = {
        "run_name": run_name,
        "episodes": config.train_episodes,
        "reward_history": reward_history,
        "loss_history": loss_history,
        "success_history": success_history,
        "average_reward": float(np.mean(reward_history)),
        "final_success_rate": float(np.mean(success_history[-config.log_interval :])),
        "epsilon_final": agent.epsilon,
        "failure_types": failure_history,
        "bootstrap": bootstrap_summary,
        "config": config.to_dict(),
    }
    return agent, training_summary


def bootstrap_agent(
    agent: DQNAgent,
    config: ArenaConfig,
    layout_seeds: tuple[int, ...],
) -> dict[str, Any]:
    demonstration_pairs: list[tuple[np.ndarray, int]] = []
    successful_episodes = 0
    episode_count = max(len(layout_seeds), config.bootstrap_expert_episodes)

    for episode in range(episode_count):
        seed = layout_seeds[episode % len(layout_seeds)]
        try:
            transitions, state_action_pairs, success = collect_expert_episode(config, seed)
        except RuntimeError:
            continue
        if not success:
            continue

        successful_episodes += 1
        demonstration_pairs.extend(state_action_pairs)
        for state, action, reward, next_state, done in transitions:
            agent.remember(state, action, reward, next_state, done)

    pretrain_losses = agent.pretrain_on_demonstrations(
        demonstration_pairs,
        epochs=config.bootstrap_pretrain_epochs,
        batch_size=config.bootstrap_batch_size,
    )
    return {
        "requested_episodes": episode_count,
        "successful_expert_episodes": successful_episodes,
        "demonstration_samples": len(demonstration_pairs),
        "pretrain_losses": pretrain_losses,
    }


def collect_expert_episode(
    config: ArenaConfig,
    layout_seed: int,
) -> tuple[
    list[tuple[np.ndarray, int, float, np.ndarray, bool]],
    list[tuple[np.ndarray, int]],
    bool,
]:
    env = ArenaEnv(config)
    state, _ = env.reset(layout_seed=layout_seed)
    transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    demonstrations: list[tuple[np.ndarray, int]] = []
    done = False
    info: dict[str, Any] = {"success": False}

    while not done:
        action = choose_expert_action(env)
        next_state, reward, done, info = env.step(action)
        transitions.append((state, action, reward, next_state, done))
        demonstrations.append((state, action))
        state = next_state

    return transitions, demonstrations, bool(info["success"])


def choose_expert_action(env: ArenaEnv) -> int:
    assert env.layout is not None and env.agent_pos is not None

    if not env.has_key:
        if env.agent_pos == env.layout.key:
            return 4
        path = shortest_action_path(env, env.agent_pos, {env.layout.key}, allow_closed_door=False)
        return path[0]

    if not env.door_open:
        if env.agent_pos.manhattan(env.layout.door) == 1:
            return 5
        targets = {
            candidate
            for candidate in adjacent_positions(env.layout.door)
            if is_traversable(env, candidate, allow_closed_door=False)
        }
        path = shortest_action_path(env, env.agent_pos, targets, allow_closed_door=False)
        return path[0]

    path = shortest_action_path(env, env.agent_pos, {env.layout.goal}, allow_closed_door=True)
    return path[0]


def shortest_action_path(
    env: ArenaEnv,
    start: Position,
    targets: set[Position],
    *,
    allow_closed_door: bool,
) -> list[int]:
    queue: list[tuple[Position, list[int]]] = [(start, [])]
    visited = {start}

    while queue:
        position, actions = queue.pop(0)
        if position in targets:
            return actions

        for action, (dx, dy) in env.ACTIONS.items():
            candidate = position.moved(dx, dy)
            if candidate in visited:
                continue
            if not is_traversable(env, candidate, allow_closed_door=allow_closed_door):
                continue
            visited.add(candidate)
            queue.append((candidate, actions + [action]))

    raise RuntimeError("Expert could not find a valid path in the arena.")


def adjacent_positions(position: Position) -> set[Position]:
    return {
        position.moved(0, -1),
        position.moved(0, 1),
        position.moved(-1, 0),
        position.moved(1, 0),
    }


def is_traversable(
    env: ArenaEnv,
    position: Position,
    *,
    allow_closed_door: bool,
) -> bool:
    assert env.layout is not None

    if not (0 <= position.x < env.grid_size and 0 <= position.y < env.grid_size):
        return False
    if position in env.layout.structural_walls or position in env.layout.obstacles:
        return False
    if position == env.layout.door and not (env.door_open or allow_closed_door):
        return False
    return True


def run_experiment_suite(
    base_config: ArenaConfig,
    results_path: Path,
    include_chaos_sweep: bool = True,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    static_train_config = replace(
        base_config,
        dynamic_goal=False,
        chaos_level=0.0,
        add_dynamic_obstacle=False,
    )
    dynamic_train_config = replace(
        base_config,
        dynamic_goal=True,
        chaos_level=base_config.chaos_level,
        add_dynamic_obstacle=True,
    )

    static_agent, static_train_summary = train_agent(static_train_config, run_name="static_agent")
    dynamic_agent, dynamic_train_summary = train_agent(dynamic_train_config, run_name="dynamic_agent")

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        static_agent.save(model_dir / "static_agent.pt")
        dynamic_agent.save(model_dir / "dynamic_agent.pt")

    static_seen_metrics = evaluate_agent(
        static_agent,
        static_train_config,
        episodes=base_config.eval_episodes,
        layout_seeds=base_config.seen_layout_seeds,
        dynamic_goal=False,
        chaos_level=0.0,
        add_dynamic_obstacle=False,
    )
    static_to_dynamic_metrics = evaluate_agent(
        static_agent,
        dynamic_train_config,
        episodes=base_config.eval_episodes,
        layout_seeds=base_config.seen_layout_seeds,
        dynamic_goal=True,
        chaos_level=base_config.chaos_level,
        add_dynamic_obstacle=True,
    )

    dynamic_seen_metrics = evaluate_agent(
        dynamic_agent,
        dynamic_train_config,
        episodes=base_config.eval_episodes,
        layout_seeds=base_config.seen_layout_seeds,
    )
    dynamic_unseen_metrics = evaluate_agent(
        dynamic_agent,
        dynamic_train_config,
        episodes=base_config.eval_episodes,
        layout_seeds=base_config.unseen_layout_seeds,
    )

    results: dict[str, Any] = {
        "metadata": {
            "project": "Agent Arena: Dynamic Facility Operations",
            "base_config": base_config.to_dict(),
        },
        "training_runs": {
            "static_agent": static_train_summary,
            "dynamic_agent": dynamic_train_summary,
        },
        "experiments": {
            "static_vs_dynamic": {
                "static_eval": static_seen_metrics,
                "dynamic_eval": static_to_dynamic_metrics,
                "robustness": build_robustness_report(
                    static_seen_metrics,
                    static_to_dynamic_metrics,
                ),
            },
            "seen_vs_unseen_layout": {
                "seen_eval": dynamic_seen_metrics,
                "unseen_eval": dynamic_unseen_metrics,
                "generalization": build_generalization_report(
                    dynamic_seen_metrics,
                    dynamic_unseen_metrics,
                ),
            },
            "chaos_level_impact": [],
        },
    }

    if include_chaos_sweep:
        sweep_episodes = max(10, base_config.train_episodes // 2)
        for chaos_level in base_config.chaos_sweep:
            sweep_config = replace(
                base_config,
                chaos_level=chaos_level,
                dynamic_goal=True,
                add_dynamic_obstacle=chaos_level > 0.0,
                train_episodes=sweep_episodes,
            )
            sweep_agent, sweep_train_summary = train_agent(
                sweep_config,
                run_name=f"chaos_{chaos_level:.2f}",
            )
            if model_dir is not None:
                sweep_agent.save(model_dir / f"chaos_{chaos_level:.2f}.pt")
            sweep_eval = evaluate_agent(
                sweep_agent,
                sweep_config,
                episodes=base_config.eval_episodes,
                layout_seeds=base_config.unseen_layout_seeds,
            )
            results["experiments"]["chaos_level_impact"].append(
                {
                    "chaos_level": chaos_level,
                    "train_summary": sweep_train_summary,
                    "eval_summary": sweep_eval,
                }
            )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agents in Agent Arena.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("experiment_results.json"),
        help="Where to write the structured experiment results JSON.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override train episodes.")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count.",
    )
    parser.add_argument("--grid-size", type=int, default=None, help="Override grid size.")
    parser.add_argument("--chaos-level", type=float, default=None, help="Override chaos level.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--skip-chaos-sweep",
        action="store_true",
        help="Skip retraining agents for the chaos impact experiment.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller experiment suite for smoke testing.",
    )
    parser.add_argument(
        "--save-models-dir",
        type=Path,
        default=None,
        help="Optional directory for saving trained model checkpoints.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ArenaConfig:
    config = ArenaConfig()

    if args.quick:
        config = replace(config, train_episodes=120, eval_episodes=8, log_interval=20)

    if args.episodes is not None:
        config = replace(config, train_episodes=args.episodes)
    if args.eval_episodes is not None:
        config = replace(config, eval_episodes=args.eval_episodes)
    if args.grid_size is not None:
        config = replace(config, grid_size=args.grid_size)
    if args.chaos_level is not None:
        config = replace(config, chaos_level=args.chaos_level)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    return config


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    results = run_experiment_suite(
        config,
        results_path=args.results_path,
        include_chaos_sweep=not args.skip_chaos_sweep,
        model_dir=args.save_models_dir,
    )

    robustness = results["experiments"]["static_vs_dynamic"]["robustness"]
    generalization = results["experiments"]["seen_vs_unseen_layout"]["generalization"]
    print("\nExperiment summary")
    print(
        f"Static -> Dynamic drop: {robustness['performance_drop']:.3f} "
        f"(before={robustness['before_dynamic_success_rate']:.3f}, "
        f"after={robustness['after_dynamic_success_rate']:.3f})"
    )
    print(
        f"Seen -> Unseen gap: {generalization['generalization_gap']:.3f} "
        f"(seen={generalization['seen_success_rate']:.3f}, "
        f"unseen={generalization['unseen_success_rate']:.3f})"
    )
    print(f"Saved structured results to {args.results_path}")


if __name__ == "__main__":
    main()
