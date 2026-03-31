from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

# In a real deployed PyPI module, this would be `from my_env.models import ...`
from my_env.models import Action, EnvResult
from my_env.server.my_environment import SupportEnvironment

app = FastAPI(title="OpenEnv - Customer Support Environment API")
env = SupportEnvironment()

class SessionRequest(BaseModel):
    session_id: str
    task_id: Optional[str] = None

@app.post("/reset", response_model=EnvResult)
def reset_env(req: SessionRequest):
    """Initializes a new simulated environment interaction loop."""
    return env.reset(req.session_id, task_id=req.task_id)

@app.post("/step/{session_id}", response_model=EnvResult)
def step_env(session_id: str, action: Action):
    """Receives the LLM's selected tool action and processes it."""
    try:
        return env.step(action, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/health")
def health_check():
    return {"status": "ok"}
