"""
OpenEnv-compliant Pydantic models for the Customer Support environment.
Inherits from openenv.core base classes (Action, Observation, State).
openenv.core.Message is a TypedDict: {"role": str, "content": str}
"""
from openenv.core import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
    Message,  # TypedDict, not Pydantic
)
from pydantic import Field
from typing import List, Optional, Dict, Any


class SupportAction(BaseAction):
    """
    Action submitted by the agent in the Customer Support environment.
    Contains the raw tool-call string or final response text.
    """
    message: str = ""


class SupportObservation(BaseObservation):
    """
    Observation returned after each step.
    Inherits `done` (bool) and `reward` (float|None) from openenv.core.Observation.
    """
    prompt: str = ""
    messages: List[Dict[str, str]] = Field(default_factory=list)


class SupportState(BaseState):
    """
    Internal episode state.
    Inherits `episode_id` (str|None) and `step_count` (int) from openenv.core.State.
    """
    task_id: str = ""
    max_steps: int = 8
    tools_used: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
