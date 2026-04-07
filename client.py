from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import AgentArenaAction, AgentArenaObservation, AgentArenaState


class AgentArenaEnv(
    EnvClient[AgentArenaAction, AgentArenaObservation, AgentArenaState]
):
    """OpenEnv client for the Dynamic Facility Operations environment."""

    def _step_payload(self, action: AgentArenaAction) -> Dict:
        return {"action": action.action.value}

    def _parse_result(self, payload: Dict) -> StepResult[AgentArenaObservation]:
        observation = AgentArenaObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AgentArenaState:
        return AgentArenaState(**payload)

