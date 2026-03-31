import re
from typing import List, Dict, Any

def extract_action(text: str) -> str:
    match = re.search(r'\[(.*?)\]', text)
    return f"[{match.group(1)}]" if match else ""

def reward_grader_score(completions, **kwargs) -> List[float]:
    """Official task success score (0.0 - 1.0)"""
    return [float(s) for s in kwargs.get("grader_score", [0.0]*len(completions))]

def reward_step_progress(completions, **kwargs) -> List[float]:
    """Sum of all partial tool-call rewards from the server."""
    rewards = kwargs.get("final_reward", [0.0]*len(completions))
    return [float(r) for r in rewards]

def reward_format(completions, **kwargs) -> List[float]:
    """Syntax check."""
    return [1.0 if extract_action(c) else 0.0 for c in completions]

def reward_conciseness(completions, **kwargs) -> List[float]:
    """Penalize excessively long responses (CSA should be efficient)."""
    rewards = []
    for c in completions:
        # Ideal length for a tool call + brief intro is < 150 chars
        if len(c) > 250:
            rewards.append(-0.5)
        elif len(c) > 150:
            rewards.append(-0.2)
        else:
            rewards.append(0.5) # Bonus for efficiency
    return rewards

def reward_repetition(completions, **kwargs) -> List[float]:
    """Penalize calling the same tool twice in the same episode."""
    reward_list = []
    action_histories = kwargs.get("action_history", [[] for _ in completions])
    
    for history in action_histories:
        if not history:
            reward_list.append(0.0)
            continue
        # Check for duplicates in the list of tool names
        if len(set(history)) < len(history):
            reward_list.append(-2.0) # High penalty for loops
        else:
            reward_list.append(0.5) # Bonus for linear progress
    return reward_list

def reward_politeness(completions, **kwargs) -> List[float]:
    """Reward specifically for using the 'respond' tool at the end of the chain."""
    reward_list = []
    action_histories = kwargs.get("action_history", [[] for _ in completions])
    
    for history in action_histories:
        if not history:
            reward_list.append(0.0)
            continue
        # If the last tool used in the episode was 'respond', they finished correctly
        if history[-1] == "respond":
            reward_list.append(1.0)
        else:
            reward_list.append(0.0)
    return reward_list
