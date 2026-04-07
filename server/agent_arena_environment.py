from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from agent_arena.config import ArenaConfig
from agent_arena.env.arena_env import ArenaEnv
from agent_arena.openenv.grader import grade_episode, normalized_step_reward, task_summary
from agent_arena.openenv.task_definitions import (
    ACTION_CONTEXT,
    TaskDefinition,
    build_task_config,
    build_curriculum_config,
    curriculum_summary,
    DIFFICULTY_BUCKET_TO_TASK_ID,
    get_task_definition,
    infer_difficulty_bucket,
    list_task_definitions,
)
from models import AgentArenaAction, AgentArenaActionType, AgentArenaObservation, AgentArenaState


ACTION_TO_ID = {
    AgentArenaActionType.UP: 0,
    AgentArenaActionType.DOWN: 1,
    AgentArenaActionType.LEFT: 2,
    AgentArenaActionType.RIGHT: 3,
    AgentArenaActionType.PICK_BADGE: 4,
    AgentArenaActionType.OPEN_GATE: 5,
}


class AgentArenaEnvironment(Environment[AgentArenaAction, AgentArenaObservation, AgentArenaState]):
    """
    OpenEnv-compatible real-world facility operations environment.

    The environment wraps the existing dynamic grid simulator but presents it
    as a warehouse/facility maintenance task with normalized scoring and
    explicit easy/medium/hard tasks required by the hackathon checklist.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self.mcp_server = self._build_mcp_server()
        self._task = get_task_definition("easy_facility_reset")
        self._env = ArenaEnv(build_task_config(self._task.task_id))
        self._difficulty_scale: float | None = None
        self._config_snapshot = self._env.get_config_snapshot()
        self._state = AgentArenaState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            difficulty_scale=None,
            status="initialized",
            status_message="Environment initialized.",
            session_active=False,
            success_threshold=self._task.success_threshold,
            current_prompt=self._task.prompt,
            config_snapshot=self._config_snapshot,
        )
        self._score = 0.0
        self._event_log: list[str] = []
        self._last_status = "Environment initialized."
        self._readme_content = self._load_readme()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> AgentArenaObservation:
        self._task, self._difficulty_scale, config = self._resolve_task_and_config(
            seed=seed,
            **kwargs,
        )
        self._env = ArenaEnv(config)
        self._config_snapshot = self._env.get_config_snapshot()

        layout_seed = kwargs.get("layout_seed", seed)
        observation, info = self._env.reset(layout_seed=layout_seed)

        self._score = 0.0
        self._event_log = [f"Task loaded: {self._task.title}"]
        self._last_status = "Collect the access badge to begin the facility run."
        self._state = AgentArenaState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            task_title=self._task.title,
            difficulty_scale=self._difficulty_scale,
            layout_seed=info["layout_seed"],
            has_badge=False,
            gate_open=False,
            dynamic_event_triggered=False,
            score=0.0,
            status="ready",
            status_message=self._last_status,
            session_active=True,
            success_threshold=self._task.success_threshold,
            current_prompt=self._task.prompt,
            config_snapshot=self._config_snapshot,
            failure_type=None,
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            status=self._last_status,
            observation_vector=observation,
            info=info,
        )

    def step(
        self,
        action: AgentArenaAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AgentArenaObservation:
        del timeout_s, kwargs

        action_id = ACTION_TO_ID[action.action]
        previous_score = self._score
        observation_vector, _, done, info = self._env.step(action_id)

        grade = grade_episode(
            has_badge=info["has_key"],
            gate_open=info["door_open"],
            success=info["success"],
            steps_taken=info["steps_taken"],
            max_steps=self._env.config.max_steps,
        )
        self._score = grade.score

        if info["picked_key"]:
            self._event_log.append("Access badge collected.")
            self._last_status = "Badge acquired. Proceed to the safety gate."
        elif info["opened_door"]:
            self._event_log.append("Safety gate unlocked.")
            self._last_status = "Gate open. Continue to the service checkpoint."
        elif info["dynamic_change"]:
            self._event_log.append("Control rerouted the checkpoint due to a facility disruption.")
            self._last_status = "Checkpoint changed. Re-plan the route immediately."
        elif info["success"]:
            self._event_log.append("Mission completed successfully.")
            self._last_status = "Checkpoint reached. Facility task completed."
        elif info["invalid_action"]:
            self._last_status = "Invalid action for the current facility state."
        else:
            self._last_status = self._default_status_message(info)

        reward = normalized_step_reward(
            previous_score=previous_score,
            current_score=self._score,
            invalid_action=info["invalid_action"],
        )

        failure_type = info.get("failure_type")
        self._state = AgentArenaState(
            episode_id=self._state.episode_id,
            step_count=info["steps_taken"],
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            task_title=self._task.title,
            difficulty_scale=self._difficulty_scale,
            layout_seed=info["layout_seed"],
            has_badge=info["has_key"],
            gate_open=info["door_open"],
            dynamic_event_triggered=info["goal_shifted"] or info["dynamic_obstacle_added"],
            score=self._score,
            status="success" if info["success"] else ("failed" if done else "in_progress"),
            status_message=self._last_status,
            session_active=True,
            success_threshold=self._task.success_threshold,
            current_prompt=self._task.prompt,
            config_snapshot=self._env.get_config_snapshot(),
            failure_type=failure_type,
        )
        return self._build_observation(
            reward=reward,
            done=done,
            status=self._last_status,
            observation_vector=observation_vector,
            info=info,
            failure_type=failure_type,
            grade_breakdown=grade.breakdown,
        )

    @property
    def state(self) -> AgentArenaState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        task_list = "\n".join(
            (
                f"- {task.task_id} ({task.difficulty}): {task.prompt} "
                f"Success criteria: {task.success_criteria}"
            )
            for task in list_task_definitions()
        )
        description = (
            "Dynamic Facility Operations is an OpenEnv simulation for evaluating how an "
            "autonomous maintenance agent adapts when mission targets move and disruptions "
            "appear mid-execution.\n\nAvailable tasks:\n"
            f"{task_list}\n\n"
            "Reset accepts either a task_id or a curriculum difficulty_scale in [0, 1] "
            "to interpolate chaos, reroute timing, and step budget."
        )
        return EnvironmentMetadata(
            name="agent_arena_dynamic_ops",
            description=description,
            readme_content=self._readme_content,
            version="1.0.0",
            author="TechWizard9999",
            documentation_url="https://github.com/TechWizard9999/Agent_Arena",
        )

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        status: str,
        observation_vector,
        info: dict[str, object],
        failure_type: str | None = None,
        grade_breakdown: dict[str, float] | None = None,
    ) -> AgentArenaObservation:
        del observation_vector
        layout = self._env.layout
        if layout is None or self._env.agent_pos is None:
            raise RuntimeError("Underlying environment must be initialized before building observation.")

        legal_actions = self._legal_actions()
        config_snapshot = self._env.get_config_snapshot()
        metadata = {
            "layout_seed": info["layout_seed"],
            "failure_type": failure_type,
            "grade_breakdown": grade_breakdown or {},
            "task": task_summary(self._task),
            "config_snapshot": config_snapshot,
            "action_context": dict(ACTION_CONTEXT),
        }
        if self._difficulty_scale is not None:
            metadata["curriculum"] = curriculum_summary(self._difficulty_scale)
        return AgentArenaObservation(
            task_id=self._task.task_id,
            task_title=self._task.title,
            task_prompt=self._task.prompt,
            difficulty=self._task.difficulty,
            difficulty_scale=self._difficulty_scale,
            status=status,
            success_criteria=self._task.success_criteria,
            evaluation_dimensions=list(self._task.evaluation_dimensions),
            expected_baseline_score_range=list(self._task.expected_baseline_score_range),
            success_threshold=self._task.success_threshold,
            grid_size=self._env.grid_size,
            grid_rows=self._env.render().splitlines(),
            legal_actions=legal_actions,
            agent_position=[self._env.agent_pos.x, self._env.agent_pos.y],
            badge_position=[layout.key.x, layout.key.y],
            gate_position=[layout.door.x, layout.door.y],
            checkpoint_position=[layout.goal.x, layout.goal.y],
            has_badge=self._env.has_key,
            gate_open=self._env.door_open,
            dynamic_event_triggered=bool(info["goal_shifted"] or info["dynamic_obstacle_added"]),
            score=self._score,
            event_log=self._event_log[-5:],
            action_descriptions=dict(ACTION_CONTEXT),
            config_snapshot=config_snapshot,
            steps_remaining=max(0, self._env.config.max_steps - int(info["steps_taken"])),
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _legal_actions(self) -> list[str]:
        legal = ["up", "down", "left", "right"]
        if self._env.layout is None or self._env.agent_pos is None:
            return legal

        if not self._env.has_key and self._env.agent_pos == self._env.layout.key:
            legal.append("pick_badge")
        if (
            self._env.has_key
            and not self._env.door_open
            and self._env.agent_pos.manhattan(self._env.layout.door) == 1
        ):
            legal.append("open_gate")
        return legal

    def _default_status_message(self, info: dict[str, object]) -> str:
        if not bool(info["has_key"]):
            return "Navigate to the access badge."
        if not bool(info["door_open"]):
            return "Move next to the safety gate and open it."
        return "Proceed to the current service checkpoint."

    def _load_readme(self) -> str:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        if readme_path.exists():
            return readme_path.read_text(encoding="utf-8")
        return "Agent Arena Dynamic Facility Operations OpenEnv."

    def _resolve_task_and_config(
        self,
        *,
        seed: int | None,
        **kwargs: Any,
    ) -> tuple[TaskDefinition, float | None, ArenaConfig]:
        difficulty_raw = kwargs.get("difficulty_scale", kwargs.get("difficulty"))
        if difficulty_raw is not None:
            difficulty_scale = max(0.0, min(1.0, float(difficulty_raw)))
            bucket = infer_difficulty_bucket(difficulty_scale)
            task = get_task_definition(DIFFICULTY_BUCKET_TO_TASK_ID[bucket])
            config = build_curriculum_config(difficulty_scale)
        else:
            task_id = str(kwargs.get("task_id", self._task.task_id))
            task = get_task_definition(task_id)
            difficulty_scale = None
            config = build_task_config(task.task_id)

        if seed is not None:
            config.seed = seed
        return task, difficulty_scale, config

    def _build_mcp_server(self) -> FastMCP:
        mcp = FastMCP("agent_arena_dynamic_ops")

        @mcp.tool
        def list_tasks() -> list[dict[str, object]]:
            """Return all benchmark tasks with success criteria and score expectations."""
            return [task_summary(task) for task in list_task_definitions()]

        @mcp.tool
        def describe_task(task_id: str) -> dict[str, object]:
            """Describe one task, including actions and what the evaluator is measuring."""
            task = get_task_definition(task_id)
            return {
                **task_summary(task),
                "action_descriptions": dict(ACTION_CONTEXT),
            }

        @mcp.tool
        def describe_controls() -> dict[str, object]:
            """Explain the action space in facility-operations terms."""
            return {
                "action_descriptions": dict(ACTION_CONTEXT),
                "sequence_requirement": "Collect the badge, open the gate, then reach the checkpoint.",
            }

        @mcp.tool
        def describe_curriculum(difficulty_scale: float) -> dict[str, object]:
            """Explain how interpolated difficulty modifies chaos, timing, and obstacles."""
            return curriculum_summary(difficulty_scale)

        @mcp.tool
        def current_session_snapshot() -> dict[str, object]:
            """Return the active session state and task metadata for inspection."""
            return {
                "state": self._state.model_dump(),
                "task": task_summary(self._task),
                "action_descriptions": dict(ACTION_CONTEXT),
            }

        return mcp
