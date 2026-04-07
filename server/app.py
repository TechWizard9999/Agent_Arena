from __future__ import annotations

import argparse

import uvicorn
from fastapi.routing import APIRoute
from openenv.core.env_server.http_server import create_app

from models import AgentArenaAction, AgentArenaObservation, AgentArenaState
from server.agent_arena_environment import AgentArenaEnvironment


app = create_app(
    AgentArenaEnvironment,
    AgentArenaAction,
    AgentArenaObservation,
    env_name="agent_arena_dynamic_ops",
    max_concurrent_envs=8,
)


def _remove_default_route(path: str) -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (isinstance(route, APIRoute) and route.path == path)
    ]


_remove_default_route("/state")
_remove_default_route("/schema")


@app.get("/state", response_model=AgentArenaState, tags=["State Management"])
def get_state() -> AgentArenaState:
    env = AgentArenaEnvironment()
    try:
        return env.state
    finally:
        env.close()


@app.get("/schema", tags=["Schema"])
def get_schema() -> dict[str, object]:
    return {
        "action": AgentArenaAction.model_json_schema(),
        "observation": AgentArenaObservation.model_json_schema(),
        "state": AgentArenaState.model_json_schema(),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
