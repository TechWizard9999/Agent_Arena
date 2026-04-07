from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class AgentArenaActionType(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    PICK_BADGE = "pick_badge"
    OPEN_GATE = "open_gate"


class AgentArenaAction(Action):
    """Typed action model for the Dynamic Facility Operations environment."""

    action: AgentArenaActionType = Field(..., description="Action to execute in the facility.")


class AgentArenaObservation(Observation):
    """Observation returned after reset() and step()."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Human-readable task title.")
    task_prompt: str = Field(..., description="Task instructions shown to the agent.")
    difficulty: str = Field(..., description="Task difficulty bucket.")
    difficulty_scale: Optional[float] = Field(
        default=None,
        description="Optional curriculum difficulty in [0, 1] when the episode is reset via interpolation.",
    )
    status: str = Field(..., description="Human-readable mission status.")
    success_criteria: str = Field(..., description="Explicit pass condition for the current task.")
    evaluation_dimensions: List[str] = Field(
        default_factory=list,
        description="Capabilities this task is intended to measure.",
    )
    expected_baseline_score_range: List[float] = Field(
        default_factory=list,
        description="Expected normalized baseline score range for this task.",
    )
    success_threshold: float = Field(..., description="Minimum normalized score considered a pass.")
    grid_size: int = Field(..., description="Square grid width/height.")
    grid_rows: List[str] = Field(
        default_factory=list,
        description="ASCII grid rows representing the current facility layout.",
    )
    legal_actions: List[str] = Field(
        default_factory=list,
        description="List of currently valid action names.",
    )
    agent_position: List[int] = Field(default_factory=list, description="Current agent position.")
    badge_position: List[int] = Field(default_factory=list, description="Access badge position.")
    gate_position: List[int] = Field(default_factory=list, description="Safety gate position.")
    checkpoint_position: List[int] = Field(
        default_factory=list,
        description="Service checkpoint position.",
    )
    has_badge: bool = Field(False, description="Whether the badge has been collected.")
    gate_open: bool = Field(False, description="Whether the safety gate is open.")
    dynamic_event_triggered: bool = Field(
        False,
        description="Whether the reroute/disruption event has happened this episode.",
    )
    score: float = Field(0.0, description="Current normalized grader score in [0, 1].")
    event_log: List[str] = Field(
        default_factory=list,
        description="Recent environment events and milestone messages.",
    )
    action_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Facility-context explanation for each action.",
    )
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Active environment configuration for the episode.",
    )
    steps_remaining: int = Field(..., description="Remaining step budget.")


class AgentArenaState(State):
    """Detailed episode state exposed via state()."""

    task_id: str = Field(default="easy_facility_reset", description="Current task identifier.")
    task_title: str = Field(default="Easy Facility Reset", description="Current task title.")
    difficulty: str = Field(default="easy", description="Difficulty bucket.")
    difficulty_scale: Optional[float] = Field(
        default=None,
        description="Optional curriculum difficulty in [0, 1].",
    )
    layout_seed: Optional[int] = Field(default=None, description="Seed used for the layout.")
    has_badge: bool = Field(default=False, description="Whether the badge has been collected.")
    gate_open: bool = Field(default=False, description="Whether the gate has been opened.")
    dynamic_event_triggered: bool = Field(
        default=False,
        description="Whether the dynamic reroute/disruption event has triggered.",
    )
    score: float = Field(default=0.0, description="Current normalized task score.")
    status: str = Field(default="ready", description="High-level environment status string.")
    status_message: str = Field(
        default="Environment initialized.",
        description="Latest human-readable guidance or outcome message.",
    )
    session_active: bool = Field(
        default=False,
        description="Whether an HTTP or WebSocket session has reset the episode and is currently active.",
    )
    success_threshold: float = Field(
        default=0.9,
        description="Minimum normalized score considered a pass for this task.",
    )
    current_prompt: str = Field(default="", description="Task instructions for the active episode.")
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Active environment configuration for the current episode.",
    )
    failure_type: Optional[str] = Field(
        default=None,
        description="Failure label set when an episode ends unsuccessfully.",
    )
