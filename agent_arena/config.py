from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ArenaConfig:
    """Central configuration for training, evaluation, and environment behavior."""

    grid_size: int = 5
    dynamic_goal: bool = True
    dynamic_change_step: int = 10
    chaos_level: float = 0.3
    add_dynamic_obstacle: bool = True
    static_obstacle_count: int | None = None
    max_steps: int = 40
    step_penalty: int = -1
    invalid_action_penalty: int = -5
    key_reward: int = 10
    door_reward: int = 10
    goal_reward: int = 50
    progress_reward_scale: float = 1.0
    interaction_zone_bonus: float = 2.0
    seed: int = 7
    loop_detection_threshold: int = 6
    train_episodes: int = 350
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.992
    replay_buffer_capacity: int = 10_000
    target_update_frequency: int = 10
    hidden_dim: int = 128
    use_expert_bootstrap: bool = True
    bootstrap_expert_episodes: int = 16
    bootstrap_pretrain_epochs: int = 6
    bootstrap_batch_size: int = 64
    eval_episodes: int = 16
    seen_layout_seeds: tuple[int, ...] = (11, 17, 23, 31, 43, 59, 67, 79)
    unseen_layout_seeds: tuple[int, ...] = (101, 113, 127, 131, 149, 157, 173, 181)
    chaos_sweep: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    render_sleep: float = 0.35
    log_interval: int = 25

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_CONFIG = ArenaConfig()
