import re
from typing import List, Dict, Any

def extract_action(text: str) -> str:
    """Extracts the tool name from the bracketed action."""
    match = re.search(r'\[(.*?)\]', text)
    if match:
        inner = match.group(1).strip()
        # Parse out just the function name (e.g., "get_order" from "get_order('123')")
        return inner.split('(')[0].strip()
    return ""

def reward_format(completions, **kwargs) -> List[float]:
    """Penalize zero-tool responses instead of giving 0, softer +0.5 reward for compliance."""
    rewards = []
    for c in completions:
        action = extract_action(c)
        rewards.append(0.5 if action else -0.5)
    return rewards

def reward_conciseness(completions, **kwargs) -> List[float]:
    """Continuous reward scaling for efficiency (better for PPO)."""
    rewards = []
    for c in completions:
        length = len(c)
        # Bounded between -0.5 and 0.5 depending on length
        reward = max(-0.5, min(0.5, (150 - length) / 300))
        rewards.append(reward)
    return rewards

def reward_repetition(completions, **kwargs) -> List[float]:
    """Softer penalty for repetitive histories to avoid crashing PPO gradients."""
    rewards = []
    histories = kwargs.get("action_history", [[] for _ in completions])

    for h in histories:
        if len(h) == 0:
            rewards.append(0.0)
            continue
        duplicates = len(h) - len(set(h))
        rewards.append(-0.5 * duplicates)
    return rewards

def reward_step_efficiency(completions, **kwargs) -> List[float]:
    """Penalize excessive meandering logic steps."""
    rewards = []
    step_counts = kwargs.get("step_count", [0]*len(completions))

    for steps in step_counts:
        rewards.append(-0.1 * steps)
    return rewards

def reward_invalid_action(completions, **kwargs) -> List[float]:
    """Penalize hallucinated tools not in the environment registry."""
    rewards = []
    valid_actions_list = kwargs.get("valid_actions", [])
    
    # Fallback to defaults if env hasn't passed them yet
    all_valid_tools = [
        "get_order", "track_shipment", "validate_return", "initiate_refund", 
        "cancel_order", "update_address", "check_delivery_slot", "reschedule_delivery", 
        "investigate_missing", "ask_proof", "validate_coupon", "reset_password", 
        "escalate_to_human", "respond"
    ]

    for i, c in enumerate(completions):
        action = extract_action(c)
        valid_actions = valid_actions_list[i] if i < len(valid_actions_list) else all_valid_tools

        if action and action not in valid_actions:
            rewards.append(-1.0)
        else:
            rewards.append(0.2)
    return rewards

def reward_task_success(completions, **kwargs) -> List[float]:
    """Amplify the grader baseline OpenEnv score."""
    scores = kwargs.get("grader_score", [0.0]*len(completions))
    return [2.0 * float(s) for s in scores]

def reward_step_progress(completions, **kwargs) -> List[float]:
    """Refined partial progress reward from the OpenEnv."""
    rewards = kwargs.get("final_reward", [0.0]*len(completions))
    return [0.5 * float(r) for r in rewards]

def reward_completion(completions, **kwargs) -> List[float]:
    """Improved politeness/wrap-up reward expecting 'respond' token."""
    rewards = []
    histories = kwargs.get("action_history", [[] for _ in completions])

    for h in histories:
        if len(h) == 0:
            rewards.append(0.0)
        elif h[-1] == "respond":
            rewards.append(1.0)
        else:
            rewards.append(-0.3)
    return rewards

def total_reward(completions, **kwargs):
    """
    Combined Reward function strictly scaled for PPO safety.
    Prioritizes Ground Truth Success > Efficiency > Format compliance
    """
    r = []

    r1 = reward_task_success(completions, **kwargs)
    r2 = reward_step_progress(completions, **kwargs)
    r3 = reward_format(completions, **kwargs)
    r4 = reward_conciseness(completions, **kwargs)
    r5 = reward_repetition(completions, **kwargs)
    r6 = reward_step_efficiency(completions, **kwargs)
    r7 = reward_invalid_action(completions, **kwargs)
    r8 = reward_completion(completions, **kwargs)

    for i in range(len(completions)):
        total = (
            r1[i] * 2.0 +   # Success is paramount
            r2[i] * 1.0 +
            r3[i] * 0.5 +
            r4[i] * 0.5 +
            r5[i] * 0.7 +
            r6[i] * 1.0 +
            r7[i] * 1.5 +
            r8[i] * 1.0
        )
        r.append(total)

    return r
