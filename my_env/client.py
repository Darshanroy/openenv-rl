import requests
import uuid
from typing import Optional
from .models import Action, EnvResult

class SupportEnvClient:
    """
    Client connector bridging the RL loop (Training) 
    with the REST API (Environment).
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        
    def reset(self, task_id: Optional[str] = None) -> EnvResult:
        # Generate unique session ID per training episode
        self.session_id = str(uuid.uuid4())
        
        payload = {"session_id": self.session_id}
        if task_id:
            payload["task_id"] = task_id
            
        response = requests.post(f"{self.base_url}/reset", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return EnvResult(**data)

    def step(self, action: Action) -> EnvResult:
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")
            
        # Pydantic handles serialization automatically when dict() is called
        response = requests.post(f"{self.base_url}/step/{self.session_id}", json=action.dict())
        response.raise_for_status()
        
        data = response.json()
        return EnvResult(**data)
        
    def close(self):
        # We don't hold persistent sockets in this REST mock,
        # but the OpenEnv standard often expects a close method.
        pass
