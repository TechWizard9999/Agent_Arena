from __future__ import annotations

import argparse
from dataclasses import replace
import time
from pathlib import Path

import torch

from agent_arena.agent.agent import DQNAgent
from agent_arena.config import ArenaConfig
from agent_arena.env.arena_env import ArenaEnv
from agent_arena.trainer.train import choose_expert_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize arena rollouts in the terminal.")
    parser.add_argument(
        "--policy",
        choices=("trained", "expert", "random"),
        default="expert",
        help="Which controller to use for the rollout.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Checkpoint path for a trained DQN policy.",
    )
    parser.add_argument("--layout-seed", type=int, default=11, help="Deterministic layout seed.")
    parser.add_argument("--grid-size", type=int, default=None, help="Optional grid size override.")
    parser.add_argument(
        "--dynamic-goal",
        action="store_true",
        help="Enable mid-episode goal relocation.",
    )
    parser.add_argument(
        "--chaos-level",
        type=float,
        default=None,
        help="Override chaos level for inserted obstacles.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds to wait between frames.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to visualize sequentially.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between frames.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ArenaConfig:
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        config = ArenaConfig(**checkpoint_config)
    else:
        config = ArenaConfig()

    if args.grid_size is not None:
        config = replace(config, grid_size=args.grid_size)
    if args.chaos_level is not None:
        config = replace(config, chaos_level=args.chaos_level)

    return replace(
        config,
        dynamic_goal=args.dynamic_goal,
        render_sleep=config.render_sleep if args.delay is None else args.delay,
    )


def load_trained_agent(config: ArenaConfig, model_path: Path) -> DQNAgent:
    env = ArenaEnv(config)
    agent = DQNAgent(env.state_size, env.action_size, config)
    agent.load(model_path)
    return agent


def render_frame(
    env: ArenaEnv,
    episode_index: int,
    step_index: int,
    action_name: str,
    reward: float,
    info: dict[str, object],
    clear_screen: bool,
) -> None:
    if clear_screen:
        print("\033[2J\033[H", end="")

    print(f"Episode {episode_index}")
    print(
        f"Step {step_index} | action={action_name} | reward={reward:.2f} "
        f"| has_key={info['has_key']} | door_open={info['door_open']}"
    )
    if info["dynamic_change"]:
        print("Dynamic event: goal moved and/or obstacle inserted.")
    print(env.render())
    print()


def rollout_episode(
    env: ArenaEnv,
    policy: str,
    *,
    episode_index: int,
    agent: DQNAgent | None,
    layout_seed: int,
    clear_screen: bool,
    delay: float,
) -> dict[str, object]:
    state, info = env.reset(layout_seed=layout_seed)
    print(f"Starting episode {episode_index} on layout seed {layout_seed}")
    print(env.render())
    print()
    time.sleep(delay)

    done = False
    step_index = 0
    total_reward = 0.0

    while not done:
        if policy == "trained":
            if agent is None:
                raise ValueError("A trained policy requires --model-path.")
            action = agent.select_action(state, explore=False)
        elif policy == "expert":
            action = choose_expert_action(env)
        else:
            action = env.sample_action()

        next_state, reward, done, info = env.step(action)
        step_index += 1
        total_reward += reward
        render_frame(
            env,
            episode_index=episode_index,
            step_index=step_index,
            action_name=env.ACTION_NAMES[action],
            reward=reward,
            info=info,
            clear_screen=clear_screen,
        )
        state = next_state
        time.sleep(delay)

    print(
        f"Episode {episode_index} finished | success={info['success']} "
        f"| steps={info['steps_taken']} | total_reward={total_reward:.2f}"
    )
    if not info["success"]:
        print(f"Failure type: {info.get('failure_type', 'unknown')}")
    print()
    return info


def main() -> None:
    args = parse_args()
    config = load_config(args)
    env = ArenaEnv(config)
    agent = load_trained_agent(config, args.model_path) if args.model_path else None

    if args.policy == "trained" and agent is None:
        raise ValueError("--model-path is required when --policy trained is selected.")

    for episode in range(1, args.episodes + 1):
        rollout_episode(
            env,
            policy=args.policy,
            episode_index=episode,
            agent=agent,
            layout_seed=args.layout_seed + episode - 1,
            clear_screen=not args.no_clear,
            delay=config.render_sleep,
        )


if __name__ == "__main__":
    main()
