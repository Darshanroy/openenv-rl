"""
OpenEnv-compliant FastAPI server for the Customer Support Environment.

Architectural Overview:
- Compliant Entry Points: Implements the standard OpenEnv '/state', '/step', and '/reset' via openenv.core.
- Session Management: Adds '/session/reset', '/session/step', etc. to support stateful, multi-turn 
  simulations in the local inference script and Streamlit dashboard.
"""
import uuid
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Internal imports from the environment package
from openenv.core import create_app
from my_env.models import SupportAction, SupportObservation, SupportState
from server.my_environment import SupportEnvironment


# ── Global Session Store ─────────────────────────────────────────────────────
# This dictionary holds active SupportEnvironment instances keyed by session_id.
# It allows the statless FastAPI server to maintain state for multiple concurrent users/tasks.
_sessions: Dict[str, SupportEnvironment] = {}


class SessionRequest(BaseModel):
    """Pydantic model for session initialization."""
    session_id: str
    task_id: Optional[str] = None


def create_environment():
    """
    Factory function required by openenv.core.create_app.
    Returns a fresh, default SupportEnvironment instance.
    """
    return SupportEnvironment(max_turns=8)


# ── OpenEnv Specification App ────────────────────────────────────────────────
# This initializes the standard OpenEnv endpoints that allow the 'openenv validate' 
# and 'openenv eval' tools to interact with this server.
app = create_app(
    env=create_environment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="CustomerSupport-v1",
    max_concurrent_envs=128,  # supporting hyper-parallel GRPO/rollout batches
)


# ── Custom Session-Based Routes ───────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe used by the OpenEnv validator and Hugging Face infrastructure."""
    return {"status": "ok", "env": "CustomerSupport-v1"}


@app.post("/session/reset")
def session_reset(req: SessionRequest):
    """
    Initializes or restarts a specific session.
    It maps a human-readable session_id to a fresh SupportEnvironment instance.
    """
    env = SupportEnvironment(max_turns=8)
    _sessions[req.session_id] = env
    obs = env.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/session/step/{session_id}")
def session_step(session_id: str, action: SupportAction):
    """
    Executes a single step (Action -> Reward/Observation) within an active session.
    If the episode is 'done', it cleans up the session memory.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /session/reset first.")
    env = _sessions[session_id]
    obs = env.step(action)
    if obs.done:
        _sessions.pop(session_id, None)  # Memory management: cleanup finished sessions
    return obs.model_dump()


class FeedbackRequest(BaseModel):
    """Pydantic model for RLHF feedback collection."""
    message_index: int
    feedback_type: str  # e.g. "thumbs_up", "thumbs_down"
    
@app.get("/session/state/{session_id}")
def session_state(session_id: str):
    """Returns the current internal state (variables, history) for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return _sessions[session_id].state.model_dump()


@app.post("/session/feedback/{session_id}")
def session_feedback(session_id: str, req: FeedbackRequest):
    """
    Collects RLHF feedback data for future model fine-tuning.
    It appends the feedback to a 'feedback.jsonl' file for persistence.
    """
    import json
    feedback_data = {
        "session_id": session_id,
        "message_index": req.message_index,
        "feedback_type": req.feedback_type,
        "timestamp": str(uuid.uuid4())[:8] 
    }
    
    # Persistent storage for reinforcement learning datasets
    try:
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
    except Exception as e:
        print(f"Error saving feedback: {e}")

    print(f"✅ [FEEDBACK RECEIVED] Session: {session_id} | Msg_Idx: {req.message_index} | Type: {req.feedback_type}")
    return {"status": "success", "session_id": session_id, "feedback": req.feedback_type}


def main():
    """
    Entry point for the environment server.
    Starts the Uvicorn server on port 7860 (Hugging Face default).
    """
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
