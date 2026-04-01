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
from server.my_environment import SupportEnvironment


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


class FeedbackRequest(BaseModel):
    message_index: int
    feedback_type: str  # e.g. "thumbs_up", "thumbs_down"
    
@app.get("/session/state/{session_id}")
def session_state(session_id: str):
    """Get state for a session-based episode."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return _sessions[session_id].state.model_dump()


@app.post("/session/feedback/{session_id}")
def session_feedback(session_id: str, req: FeedbackRequest):
    """Log feedback for an RLHF dataset or immediate dashboard reporting."""
    import json
    feedback_data = {
        "session_id": session_id,
        "message_index": req.message_index,
        "feedback_type": req.feedback_type,
        "timestamp": str(uuid.uuid4())[:8] # simplified for now
    }
    
    # Save to local file for training data collection
    try:
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
    except Exception as e:
        print(f"Error saving feedback: {e}")

    print(f"✅ [FEEDBACK RECEIVED] Session: {session_id} | Msg_Idx: {req.message_index} | Type: {req.feedback_type}")
    return {"status": "success", "session_id": session_id, "feedback": req.feedback_type}



def main():
    """Entry point for `openenv serve` and `[project.scripts]`."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
