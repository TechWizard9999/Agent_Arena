---
title: Agent Arena Dynamic Ops
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - robotics
  - logistics
---

# AI Agent Arena: Dynamic Facility Operations

AI Agent Arena has been converted into an **OpenEnv-compliant environment** for the Meta × PyTorch OpenEnv hackathon. Instead of presenting the benchmark as a toy key-door-goal game, this version frames the environment as a **real-world facility operations task**:

- collect an access badge
- unlock a safety gate
- reach the active service checkpoint
- adapt when the checkpoint reroutes or an aisle becomes blocked mid-run

The original PyTorch + DQN training code is still included under `agent_arena/`, but the repository root is now shaped as a deployable OpenEnv environment with manifest, typed models, server, Dockerfile, baseline inference script, and Hugging Face Space-ready README metadata.

## Why This Fits The Hackathon

This project is built around the hackathon goal of evaluating intelligence under dynamic conditions:

- **Real-world task framing**: secure facility maintenance and autonomous dispatch
- **Multi-step reasoning**: badge pickup -> gate unlock -> checkpoint reach
- **Dynamic environment**: checkpoint reroutes and new obstacles can appear
- **Three graded tasks**: easy, medium, hard
- **Normalized scoring**: task grading and OpenEnv-facing reward in `[0.0, 1.0]`
- **OpenEnv compliance**: `openenv.yaml`, FastAPI app, `/reset`, `/step`, `/state`, `/schema`, `/health`, `/mcp`, Docker deployment

## Tasks

The environment exposes three submission-ready tasks:

1. `easy_facility_reset`
   Static checkpoint, no dynamic disruption, generous step budget.
2. `medium_reroute_response`
   The checkpoint can move after a facility alert.
3. `hard_disruption_recovery`
   The checkpoint can reroute and a new blockage can appear while the mission is underway.

Each task uses deterministic layout seeds so you can report reproducible baseline scores.

## Scoring And Grading

Scores are normalized to `[0.0, 1.0]`:

- `0.30` for collecting the badge
- `0.30` for unlocking the gate
- `0.30` for reaching the checkpoint
- up to `0.10` efficiency bonus for successful fast completions

This gives a meaningful partial-credit grader while keeping the final reward/score in the range the hackathon expects.

## OpenEnv Interface

The deployed environment supports:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /metadata`
- `GET /health`
- `POST /mcp`
- `WS /ws`

Key files:

- `openenv.yaml`
- `pyproject.toml`
- `models.py`
- `server/app.py`
- `server/agent_arena_environment.py`
- `baseline_inference.py`

## Local Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate or refresh the OpenEnv lockfile:

```bash
python3 -m uv lock
```

Validate the environment directory:

```bash
openenv validate .
```

Run the server locally:

```bash
python3 -m server.app
```

Validate the running server:

```bash
openenv validate http://127.0.0.1:8000
```

## Baseline Inference

Run the reproducible direct baseline:

```bash
python3 baseline_inference.py --episodes-per-task 4
```

Run the same baseline against a running OpenEnv server:

```bash
python3 baseline_inference.py \
  --base-url http://127.0.0.1:8000 \
  --episodes-per-task 4
```

The baseline uses a deterministic expert-style controller and reports normalized per-task scores in JSON.

## Hugging Face Spaces

This repo is configured for Docker-based Hugging Face Spaces deployment.

Required repo files are already present:

- `openenv.yaml`
- `Dockerfile` (repo root)
- root README frontmatter for Space configuration

Recommended Space variables:

- `API_BASE_URL`
- `MODEL_NAME`

These are surfaced in environment metadata so the runtime configuration is visible to reviewers, even though the included baseline is heuristic and does not require a hosted LLM.

## RL Training and Experiments

The DQN training framework is fully functional. Run the complete experiment suite:

```bash
python -m agent_arena.trainer.train --results-path experiment_results.json
```

For a quicker smoke test:

```bash
python -m agent_arena.trainer.train --quick --results-path experiment_results.json
```

Generate plots from the results:

```bash
python -m agent_arena.plots.plot_metrics --results-path experiment_results.json --output-dir plots/
```

Visualize an expert rollout in the terminal:

```bash
python -m agent_arena.demo.visualize --dynamic-goal --episodes 3
```

Key RL files:

- `agent_arena/env/arena_env.py` — dynamic grid-world environment
- `agent_arena/agent/dqn.py` — DQN network (2 hidden layers)
- `agent_arena/agent/agent.py` — agent with target network, replay buffer, ε-greedy
- `agent_arena/trainer/train.py` — training loop with static vs dynamic experiments
- `agent_arena/evaluator/metrics.py` — success rate, robustness, generalization metrics
- `agent_arena/plots/plot_metrics.py` — reward curves, chaos sweep, comparison plots

The experiments demonstrate:

- static-trained agents fail under dynamic conditions
- dynamic-trained agents show robustness to goal rerouting
- higher chaos levels degrade success rates measurably
- generalization gap between seen and unseen layouts

## Expected Outcome

This repository now aims to satisfy the hackathon submission needs in one place:

- an OpenEnv environment that validates locally and at runtime
- a real-world task framing instead of a toy game framing
- three difficulty-tiered tasks
- normalized grading
- baseline inference
- Docker/HF Space readiness

The intended story remains the same:

> Agents trained in static environments fail under dynamic conditions.
