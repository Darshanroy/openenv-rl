"""
OpenEnv-compliant FastAPI server for the Customer Support Environment.
Uses openenv.core.create_app() for spec compliance + custom session routes
for the Streamlit dashboard (which needs stateful multi-turn episodes).
"""
import uuid
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openenv.core import create_app
from my_env.models import SupportAction, SupportObservation, SupportState
from my_env.server.my_environment import SupportEnvironment


# ── Session store for multi-turn conversations ───────────────────────────────
_sessions: Dict[str, SupportEnvironment] = {}


class SessionRequest(BaseModel):
    session_id: str
    task_id: Optional[str] = None


def create_environment():
    """Factory function that returns a fresh SupportEnvironment instance."""
    return SupportEnvironment(max_turns=8)


# Create the OpenEnv-spec app (validates with `openenv validate`)
app = create_app(
    env=create_environment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="CustomerSupport-v1",
    max_concurrent_envs=128,  # supporting hyper-parallel GRPO/rollout batches
)


# ── Custom session-based routes (for Streamlit dashboard) ────────────────────

@app.get("/health")
def health():
    """Liveness probe for the OpenEnv validator."""
    return {"status": "ok", "env": "CustomerSupport-v1"}


@app.post("/session/reset")
def session_reset(req: SessionRequest):
    """Start a new session-based episode for the dashboard."""
    env = SupportEnvironment(max_turns=8)
    _sessions[req.session_id] = env
    obs = env.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/session/step/{session_id}")
def session_step(session_id: str, action: SupportAction):
    """Step a session-based episode."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /session/reset first.")
    env = _sessions[session_id]
    obs = env.step(action)
    if obs.done:
        _sessions.pop(session_id, None)  # cleanup finished sessions
    return obs.model_dump()


@app.get("/session/state/{session_id}")
def session_state(session_id: str):
    """Get state for a session-based episode."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return _sessions[session_id].state.model_dump()


def main():
    """Entry point for `openenv serve` and `[project.scripts]`."""
    import uvicorn
    uvicorn.run("my_env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
