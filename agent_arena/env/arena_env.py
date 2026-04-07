from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from agent_arena.config import ArenaConfig
from agent_arena.env.objects import ArenaLayout, Position, Tokens


class ArenaEnv:
    """Dynamic grid-world for evaluating sequential reasoning under change."""

    ACTIONS = {
        0: (0, -1),   # up
        1: (0, 1),    # down
        2: (-1, 0),   # left
        3: (1, 0),    # right
    }

    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "pick_key",
        5: "open_door",
    }

    def __init__(self, config: ArenaConfig | None = None) -> None:
        self.config = config or ArenaConfig()
        self.grid_size = self.config.grid_size
        self.action_size = 6
        self.state_size = 14
        self.wall_x = max(1, self.grid_size // 2)
        self.rng = np.random.default_rng(self.config.seed)

        self.layout: ArenaLayout | None = None
        self.agent_pos: Position | None = None
        self.has_key = False
        self.door_open = False
        self.steps_taken = 0
        self.goal_shifted = False
        self.dynamic_obstacle_added = False
        self.invalid_actions = 0
        self.history: list[tuple[int, ...]] = []
        self.layout_seed: int | None = None
        self.key_zone_rewarded = False
        self.door_zone_rewarded = False

    def reset(self, layout_seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5 to place the full task.")

        self.layout_seed = self.config.seed if layout_seed is None else layout_seed
        self.rng = np.random.default_rng(self.layout_seed)

        self.layout = self._generate_layout()
        self.agent_pos = self.layout.agent
        self.has_key = False
        self.door_open = False
        self.steps_taken = 0
        self.goal_shifted = False
        self.dynamic_obstacle_added = False
        self.invalid_actions = 0
        self.history = [self._history_signature()]
        self.key_zone_rewarded = False
        self.door_zone_rewarded = False
        return self._get_state(), self._build_info(dynamic_change=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.layout is None or self.agent_pos is None:
            raise RuntimeError("Environment must be reset before stepping.")

        reward = float(self.config.step_penalty)
        done = False
        invalid_action = False
        picked_key = False
        opened_door = False
        dynamic_change = False
        phase_before, distance_before = self._phase_and_distance()

        if action in self.ACTIONS:
            reward, invalid_action = self._handle_move(action, reward)
        elif action == 4:
            reward, picked_key, invalid_action = self._handle_pick_key(reward)
        elif action == 5:
            reward, opened_door, invalid_action = self._handle_open_door(reward)
        else:
            invalid_action = True
            reward += self.config.invalid_action_penalty

        if action in self.ACTIONS and not invalid_action:
            phase_after, distance_after = self._phase_and_distance()
            if (
                phase_before == phase_after
                and distance_before is not None
                and distance_after is not None
            ):
                reward += self.config.progress_reward_scale * (distance_before - distance_after)

            if (
                not self.has_key
                and self.agent_pos == self.layout.key
                and not self.key_zone_rewarded
            ):
                reward += self.config.interaction_zone_bonus
                self.key_zone_rewarded = True

            if (
                self.has_key
                and not self.door_open
                and self.agent_pos.manhattan(self.layout.door) == 1
                and not self.door_zone_rewarded
            ):
                reward += self.config.interaction_zone_bonus
                self.door_zone_rewarded = True

        self.steps_taken += 1

        success = self._goal_reached()
        if not success and (
            self.config.dynamic_goal
            and not self.goal_shifted
            and self.steps_taken >= self.config.dynamic_change_step
        ):
            dynamic_change = self._trigger_dynamic_change()
            success = self._goal_reached()

        if success:
            reward += self.config.goal_reward
            done = True
        elif self.steps_taken >= self.config.max_steps:
            done = True

        self.history.append(self._history_signature())
        info = self._build_info(
            dynamic_change=dynamic_change,
            picked_key=picked_key,
            opened_door=opened_door,
            invalid_action=invalid_action,
            success=success,
        )

        if done and not success:
            info["failure_type"] = self._failure_type()

        return self._get_state(), reward, done, info

    def render(self) -> str:
        if self.layout is None or self.agent_pos is None:
            raise RuntimeError("Environment must be reset before rendering.")

        grid = [[Tokens.EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for wall in self.layout.structural_walls:
            grid[wall.y][wall.x] = Tokens.STRUCTURAL_WALL

        for obstacle in self.layout.obstacles:
            grid[obstacle.y][obstacle.x] = Tokens.OBSTACLE

        grid[self.layout.goal.y][self.layout.goal.x] = Tokens.GOAL
        if not self.has_key:
            grid[self.layout.key.y][self.layout.key.x] = Tokens.KEY

        grid[self.layout.door.y][self.layout.door.x] = (
            Tokens.OPEN_DOOR if self.door_open else Tokens.CLOSED_DOOR
        )
        grid[self.agent_pos.y][self.agent_pos.x] = Tokens.AGENT
        return "\n".join(" ".join(row) for row in grid)

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.action_size))

    def get_config_snapshot(self) -> dict[str, Any]:
        return asdict(self.config)

    def _generate_layout(self) -> ArenaLayout:
        left_cells = [Position(x, y) for x in range(self.wall_x) for y in range(self.grid_size)]
        right_cells = [
            Position(x, y)
            for x in range(self.wall_x + 1, self.grid_size)
            for y in range(self.grid_size)
        ]

        door_y = int(self.rng.integers(1, self.grid_size - 1))
        door = Position(self.wall_x, door_y)

        structural_walls = {
            Position(self.wall_x, y)
            for y in range(self.grid_size)
            if y != door_y
        }

        agent = self._sample_from(left_cells)
        key = self._sample_from([cell for cell in left_cells if cell != agent])
        goal = self._sample_from(right_cells)

        reserved = {agent, key, goal, door} | structural_walls
        obstacle_count = self._initial_obstacle_count()
        obstacles = self._sample_obstacles(obstacle_count, reserved)

        return ArenaLayout(
            agent=agent,
            key=key,
            door=door,
            goal=goal,
            structural_walls=structural_walls,
            obstacles=obstacles,
        )

    def _handle_move(self, action: int, reward: float) -> tuple[float, bool]:
        assert self.layout is not None and self.agent_pos is not None

        dx, dy = self.ACTIONS[action]
        candidate = self.agent_pos.moved(dx, dy)

        if not self._is_valid_position(candidate):
            self.invalid_actions += 1
            return reward + self.config.invalid_action_penalty, True

        if candidate == self.layout.door and not self.door_open:
            self.invalid_actions += 1
            return reward + self.config.invalid_action_penalty, True

        if candidate in self.layout.structural_walls or candidate in self.layout.obstacles:
            self.invalid_actions += 1
            return reward + self.config.invalid_action_penalty, True

        self.agent_pos = candidate
        return reward, False

    def _handle_pick_key(self, reward: float) -> tuple[float, bool, bool]:
        assert self.layout is not None and self.agent_pos is not None

        if not self.has_key and self.agent_pos == self.layout.key:
            self.has_key = True
            return reward + self.config.key_reward, True, False

        self.invalid_actions += 1
        return reward + self.config.invalid_action_penalty, False, True

    def _handle_open_door(self, reward: float) -> tuple[float, bool, bool]:
        assert self.layout is not None and self.agent_pos is not None

        adjacent = self.agent_pos.manhattan(self.layout.door) == 1
        if self.has_key and not self.door_open and adjacent:
            self.door_open = True
            return reward + self.config.door_reward, True, False

        self.invalid_actions += 1
        return reward + self.config.invalid_action_penalty, False, True

    def _trigger_dynamic_change(self) -> bool:
        assert self.layout is not None

        candidate_positions = [
            Position(x, y)
            for x in range(self.wall_x + 1, self.grid_size)
            for y in range(self.grid_size)
            if Position(x, y) not in self.layout.obstacles
            and Position(x, y) != self.layout.goal
            and Position(x, y) != self.layout.door
        ]

        if candidate_positions:
            self.layout.goal = self._sample_from(candidate_positions)
            self.goal_shifted = True

        if self.config.add_dynamic_obstacle and self.rng.random() < self.config.chaos_level:
            reserved = {
                self.agent_pos,
                self.layout.key if not self.has_key else None,
                self.layout.goal,
                self.layout.door,
            }
            reserved = {cell for cell in reserved if cell is not None}
            free_cells = [
                Position(x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                if Position(x, y) not in reserved
                and Position(x, y) not in self.layout.obstacles
                and Position(x, y) not in self.layout.structural_walls
            ]
            if free_cells:
                self.layout.obstacles.add(self._sample_from(free_cells))
                self.dynamic_obstacle_added = True

        return self.goal_shifted or self.dynamic_obstacle_added

    def _goal_reached(self) -> bool:
        assert self.layout is not None and self.agent_pos is not None
        return self.door_open and self.agent_pos == self.layout.goal

    def _get_state(self) -> np.ndarray:
        assert self.layout is not None and self.agent_pos is not None

        scale = max(1, self.grid_size - 1)
        state = np.array(
            [
                self.agent_pos.x / scale,
                self.agent_pos.y / scale,
                self.layout.key.x / scale,
                self.layout.key.y / scale,
                self.layout.door.x / scale,
                self.layout.door.y / scale,
                self.layout.goal.x / scale,
                self.layout.goal.y / scale,
                float(self.has_key),
                float(self.door_open),
                self.steps_taken / max(1, self.config.max_steps),
                float(self.goal_shifted or self.dynamic_obstacle_added),
                float(not self.has_key and self.agent_pos == self.layout.key),
                float(
                    self.has_key
                    and not self.door_open
                    and self.agent_pos.manhattan(self.layout.door) == 1
                ),
            ],
            dtype=np.float32,
        )
        return state

    def _build_info(
        self,
        dynamic_change: bool,
        picked_key: bool = False,
        opened_door: bool = False,
        invalid_action: bool = False,
        success: bool = False,
    ) -> dict[str, Any]:
        return {
            "picked_key": picked_key,
            "opened_door": opened_door,
            "invalid_action": invalid_action,
            "dynamic_change": dynamic_change,
            "goal_shifted": self.goal_shifted,
            "dynamic_obstacle_added": self.dynamic_obstacle_added,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "success": success,
            "steps_taken": self.steps_taken,
            "layout_seed": self.layout_seed,
        }

    def _failure_type(self) -> str:
        history_counts: dict[tuple[int, ...], int] = {}
        for signature in self.history:
            history_counts[signature] = history_counts.get(signature, 0) + 1

        max_repetition = max(history_counts.values(), default=0)
        if not self.has_key:
            return "did_not_pick_key"
        if self.goal_shifted and not self._goal_reached():
            return "failed_after_environment_change"
        if max_repetition >= self.config.loop_detection_threshold:
            return "got_stuck_in_loop"
        return "max_steps_exceeded"

    def _history_signature(self) -> tuple[int, ...]:
        assert self.agent_pos is not None
        return (
            self.agent_pos.x,
            self.agent_pos.y,
            int(self.has_key),
            int(self.door_open),
        )

    def _sample_from(self, positions: list[Position]) -> Position:
        index = int(self.rng.integers(0, len(positions)))
        return positions[index]

    def _sample_obstacles(
        self,
        count: int,
        reserved: set[Position],
    ) -> set[Position]:
        if count <= 0:
            return set()

        candidates = [
            Position(x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if Position(x, y) not in reserved
        ]

        if not candidates:
            return set()

        obstacle_count = min(count, len(candidates))
        sampled = self.rng.choice(candidates, size=obstacle_count, replace=False)
        return {Position(int(position.x), int(position.y)) for position in sampled}

    def _initial_obstacle_count(self) -> int:
        if self.config.static_obstacle_count is not None:
            return self.config.static_obstacle_count

        count = int(round(self.config.chaos_level * max(1, self.grid_size - 2)))
        return min(count, max(0, self.grid_size - 2))

    def _is_valid_position(self, position: Position) -> bool:
        return 0 <= position.x < self.grid_size and 0 <= position.y < self.grid_size

    def _phase_and_distance(self) -> tuple[str, int | None]:
        assert self.layout is not None and self.agent_pos is not None

        if not self.has_key:
            return "get_key", self.agent_pos.manhattan(self.layout.key)
        if not self.door_open:
            return "reach_door", max(0, self.agent_pos.manhattan(self.layout.door) - 1)
        return "reach_goal", self.agent_pos.manhattan(self.layout.goal)
