from __future__ import annotations

from enum import Enum
from typing import List, Optional

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
    task_prompt: str = Field(..., description="Task instructions shown to the agent.")
    difficulty: str = Field(..., description="Task difficulty bucket.")
    status: str = Field(..., description="Human-readable mission status.")
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
    steps_remaining: int = Field(..., description="Remaining step budget.")


class AgentArenaState(State):
    """Detailed episode state exposed via state()."""

    task_id: str = Field(default="easy_facility_reset", description="Current task identifier.")
    difficulty: str = Field(default="easy", description="Difficulty bucket.")
    layout_seed: Optional[int] = Field(default=None, description="Seed used for the layout.")
    has_badge: bool = Field(default=False, description="Whether the badge has been collected.")
    gate_open: bool = Field(default=False, description="Whether the gate has been opened.")
    dynamic_event_triggered: bool = Field(
        default=False,
        description="Whether the dynamic reroute/disruption event has triggered.",
    )
    score: float = Field(default=0.0, description="Current normalized task score.")
    status: str = Field(default="ready", description="High-level environment status string.")
    failure_type: Optional[str] = Field(
        default=None,
        description="Failure label set when an episode ends unsuccessfully.",
    )

