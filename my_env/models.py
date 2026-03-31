from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class Message(BaseModel):
    category: str  # "CUSTOMER", "AGENT", "FEEDBACK", "SYSTEM"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    prompt: str
    messages: List[Message] = Field(default_factory=list)

class Action(BaseModel):
    message: str

class State(BaseModel):
    """Internal environment state for tracking progress & history."""
    session_id: str
    current_turn: int
    max_turns: int
    history: List[Message] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    done: bool

class EnvResult(BaseModel):
    observation: Observation
    reward: Optional[float] = 0.0
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict) # For task scores & metadata
