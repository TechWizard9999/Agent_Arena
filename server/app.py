from __future__ import annotations

import argparse
import threading
from typing import Any

import uvicorn
from fastapi import Body, HTTPException
from fastapi.routing import APIRoute
from openenv.core.env_server.http_server import (
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    create_app,
    deserialize_action,
    serialize_observation,
)
from pydantic import ValidationError

from agent_arena.openenv.grader import task_summary
from agent_arena.openenv.task_definitions import (
    ACTION_CONTEXT,
    curriculum_summary,
    list_task_definitions,
)
from models import AgentArenaAction, AgentArenaObservation, AgentArenaState
from server.agent_arena_environment import AgentArenaEnvironment


app = create_app(
    AgentArenaEnvironment,
    AgentArenaAction,
    AgentArenaObservation,
    env_name="agent_arena_dynamic_ops",
    max_concurrent_envs=8,
)

_HTTP_ENV: AgentArenaEnvironment | None = None
_HTTP_ENV_LOCK = threading.RLock()


def _remove_default_route(path: str) -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (isinstance(route, APIRoute) and route.path == path)
    ]


for route_path in ("/reset", "/step", "/state", "/schema"):
    _remove_default_route(route_path)


def _close_http_env() -> None:
    global _HTTP_ENV

    if _HTTP_ENV is not None:
        _HTTP_ENV.close()
        _HTTP_ENV = None


def _empty_state() -> AgentArenaState:
    task = list_task_definitions()[0]
    return AgentArenaState(
        episode_id=None,
        step_count=0,
        task_id=task.task_id,
        task_title=task.title,
        difficulty=task.difficulty,
        difficulty_scale=None,
        layout_seed=None,
        has_badge=False,
        gate_open=False,
        dynamic_event_triggered=False,
        score=0.0,
        status="no_active_session",
        status_message="Call /reset to start an episode before requesting state or stepping.",
        session_active=False,
        success_threshold=task.success_threshold,
        current_prompt=task.prompt,
        config_snapshot={},
        failure_type=None,
    )


def shutdown_http_session() -> None:
    with _HTTP_ENV_LOCK:
        _close_http_env()


app.router.on_shutdown.append(shutdown_http_session)


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
def reset_environment(
    request: ResetRequest = Body(default_factory=ResetRequest),
) -> ResetResponse:
    """Reset the persistent HTTP session and return the first observation."""
    global _HTTP_ENV

    with _HTTP_ENV_LOCK:
        _close_http_env()
        _HTTP_ENV = AgentArenaEnvironment()
        try:
            observation = _HTTP_ENV.reset(**request.model_dump(exclude_unset=True))
        except Exception:
            _close_http_env()
            raise
        return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
def step_environment(request: StepRequest) -> StepResponse:
    """Execute one action against the persistent HTTP session."""
    with _HTTP_ENV_LOCK:
        if _HTTP_ENV is None or not _HTTP_ENV.state.session_active:
            raise HTTPException(status_code=409, detail="Call /reset before /step.")

        try:
            action = deserialize_action(request.action, AgentArenaAction)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        kwargs = request.model_dump(exclude_unset=True, exclude={"action"})
        observation = _HTTP_ENV.step(action, **kwargs)
        return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=AgentArenaState, tags=["State Management"])
def get_state() -> AgentArenaState:
    """Return the active HTTP session state, or a sentinel when no session exists."""
    with _HTTP_ENV_LOCK:
        if _HTTP_ENV is None:
            return _empty_state()
        return _HTTP_ENV.state


@app.get("/schema", tags=["Schema"])
def get_schema() -> dict[str, object]:
    """Return JSON schemas for actions, observations, state, and HTTP request bodies."""
    return {
        "action": AgentArenaAction.model_json_schema(),
        "observation": AgentArenaObservation.model_json_schema(),
        "state": AgentArenaState.model_json_schema(),
        "reset_request": ResetRequest.model_json_schema(),
        "step_request": StepRequest.model_json_schema(),
    }


@app.get("/env-info", tags=["Environment Info"])
def get_env_info() -> dict[str, Any]:
    """Expose task catalog, controls, and the active session snapshot."""
    with _HTTP_ENV_LOCK:
        current_state = None if _HTTP_ENV is None else _HTTP_ENV.state.model_dump()

    return {
        "environment": "agent_arena_dynamic_ops",
        "tasks": [task_summary(task) for task in list_task_definitions()],
        "action_descriptions": dict(ACTION_CONTEXT),
        "reset_options": {
            "task_id": [task.task_id for task in list_task_definitions()],
            "layout_seed": "Optional integer seed for deterministic layouts.",
            "difficulty_scale": "Optional float in [0, 1] for interpolated curriculum difficulty.",
        },
        "curriculum_example": curriculum_summary(0.5),
        "active_http_session": current_state,
    }


@app.get("/", tags=["Environment Info"])
def root() -> dict[str, Any]:
    """Return a simple 200 response for Space health probes and manual checks."""
    return {
        "name": "agent_arena_dynamic_ops",
        "status": "ok",
        "message": "Agent Arena Dynamic Facility Operations is running.",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
        "schema": "/schema",
    }


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.port == 7860:
        main()
    else:
        main(port=args.port)
