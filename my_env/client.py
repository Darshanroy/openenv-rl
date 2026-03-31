"""
OpenEnv-compliant client for the Customer Support Environment.
Uses session-based routes (/session/reset, /session/step, /session/state)
for stateful multi-turn conversations with the dashboard.
"""
import requests
import uuid
from typing import Optional

from .models import SupportAction, SupportObservation, SupportState


class SupportEnvClient:
    """
    Client connector bridging the RL training loop / Streamlit dashboard
    with the OpenEnv Environment server.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def reset(self, task_id: Optional[str] = None) -> SupportObservation:
        """Start a new episode. Returns the initial observation."""
        self.session_id = str(uuid.uuid4())

        payload = {"session_id": self.session_id}
        if task_id:
            payload["task_id"] = task_id

        response = requests.post(f"{self.base_url}/session/reset", json=payload)
        response.raise_for_status()
        return SupportObservation(**response.json())

    def step(self, action: SupportAction) -> SupportObservation:
        """Submit an action, receive the next observation."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")

        response = requests.post(
            f"{self.base_url}/session/step/{self.session_id}",
            json=action.model_dump(),
        )
        response.raise_for_status()
        return SupportObservation(**response.json())

    def get_state(self) -> SupportState:
        """Retrieve the current episode state."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before get_state()")
        response = requests.get(f"{self.base_url}/session/state/{self.session_id}")
        response.raise_for_status()
        return SupportState(**response.json())

    def send_feedback(self, message_index: int, feedback_type: str):
        """Send RLHF feedback for a specific completed assistant message."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before send_feedback()")
        
        payload = {"message_index": message_index, "feedback_type": feedback_type}
        try:
            requests.post(f"{self.base_url}/session/feedback/{self.session_id}", json=payload)
        except Exception:
            pass # Non-blocking on failure

    def close(self):
        """Clean up (no persistent sockets in REST mode)."""
        pass
