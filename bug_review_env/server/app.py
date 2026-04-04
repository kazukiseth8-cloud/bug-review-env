"""
FastAPI server for the Bug Review Environment.
Exposes /reset, /step, /state endpoints per OpenEnv spec.
"""

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from bug_review_env.server.environment import BugReviewEnvironment, TASK_ORDER
from bug_review_env.models import BugReviewAction, BugReviewObservation, BugReviewState

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bug Review Environment",
    description=(
        "An OpenEnv-compatible environment where an LLM agent reviews code "
        "snippets and identifies bugs. Three tasks: easy → medium → hard."
    ),
    version="1.0.0",
)

# One environment instance per server (stateless HTTP; each reset() starts fresh)
_env = BugReviewEnvironment()
_session_id: str = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Request / Response schemas (OpenEnv spec)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = "find_bug_easy"

class StepRequest(BaseModel):
    action: Dict[str, Any]

class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = {}

class StateResponse(BaseModel):
    episode_id: str
    step_count: int
    task_name: str
    attempts: int
    last_score: float

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "bug-review-env",
        "version": "1.0.0",
        "description": "OpenEnv bug review environment with 3 tasks (easy/medium/hard)",
        "tasks": TASK_ORDER,
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and return the initial observation.
    Optionally pass task_name: "find_bug_easy" | "find_bug_medium" | "find_bug_hard"
    """
    task_name = request.task_name or "find_bug_easy"
    obs = _env.reset(task_name=task_name)
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    """
    Submit an action and receive observation, reward, done, info.
    Action fields: buggy_line (int), bug_type (str), explanation (str)
    """
    try:
        action = BugReviewAction(**request.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {e}")

    obs, reward, done = _env.step(action)

    return StepResult(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info={"score": reward, "task": _env.state.task_name},
    ).model_dump()


@app.get("/state")
def state():
    """Return current episode state metadata."""
    s = _env.state
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        task_name=s.task_name,
        attempts=s.attempts,
        last_score=s.last_score,
    ).model_dump()


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions."""
    from bug_review_env.server.environment import TASKS
    return {
        name: {
            "display_name": info["display_name"],
            "instructions": info["instructions"],
        }
        for name, info in TASKS.items()
    }
