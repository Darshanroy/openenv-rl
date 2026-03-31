"""
Production-Grade Reward Functions for TRL GRPOTrainer.

Preserves the original 9-function weighted reward pipeline with PPO-stable
clipping. Adapted to work with TRL's environment_factory pattern where
reward functions receive `environments` (list of SupportToolEnv instances).

The `environments` parameter gives access to env state (tools_used, task_id,
step_count, etc.) while `**kwargs` includes `completions` (list of strings).
"""
import re
from typing import List, Dict, Any


def extract_action(text: str) -> str:
    """Extracts the tool name from the bracketed action."""
    match = re.search(r'\[(.*?)\]', text)
    if match:
        inner = match.group(1).strip()
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
    """Smooth decay penalizing excessive meandering logic steps."""
    rewards = []
    step_counts = kwargs.get("step_count", [0]*len(completions))

    for steps in step_counts:
        rewards.append(-0.05 * steps)
    return rewards


def reward_invalid_action(completions, **kwargs) -> List[float]:
    """Penalize hallucinated tools not in the environment registry and empty actions."""
    rewards = []
    valid_actions_list = kwargs.get("valid_actions", [])

    all_valid_tools = [
        "get_order", "track_shipment", "validate_return", "initiate_refund",
        "cancel_order", "update_address", "check_delivery_slot", "reschedule_delivery",
        "investigate_missing", "ask_proof", "validate_coupon", "reset_password",
        "escalate_to_human", "respond"
    ]

    for i, c in enumerate(completions):
        action = extract_action(c)
        valid_actions = valid_actions_list[i] if i < len(valid_actions_list) else all_valid_tools

        if not action:
            rewards.append(-0.3)
        elif action not in valid_actions:
            rewards.append(-1.0)
        else:
            rewards.append(0.3)
    return rewards


def reward_action_alignment(completions, **kwargs) -> List[float]:
    """State-aware correctness: Reward for picking a tool appropriate for the current intent/task."""
    rewards = []
    intents = kwargs.get("task_id", kwargs.get("intent", []))

    intent_to_tools = {
        "easy_status": ["get_order", "track_shipment", "respond"],
        "easy_payment_fail": ["get_order", "respond"],
        "easy_coupon": ["validate_coupon", "respond"],
        "easy_account": ["reset_password", "respond"],
        "easy_cancel": ["get_order", "cancel_order", "respond"],
        "medium_delay": ["get_order", "track_shipment", "respond"],
        "medium_address": ["get_order", "update_address", "respond"],
        "medium_reschedule": ["get_order", "check_delivery_slot", "reschedule_delivery", "respond"],
        "medium_return": ["get_order", "validate_return", "create_return_request", "respond"],
        "medium_double_charge": ["get_order", "initiate_refund", "respond"],
        "hard_refund": ["get_order", "validate_return", "initiate_refund", "respond"],
        "hard_damaged": ["get_order", "ask_proof", "validate_return", "initiate_refund", "respond"],
        "hard_missing": ["get_order", "track_shipment", "investigate_missing", "escalate_to_human", "respond"],
        "hard_angry": ["get_order", "track_shipment", "respond"],
        "hard_escalation": ["get_order", "escalate_to_human", "respond"],
    }

    for i, c in enumerate(completions):
        action = extract_action(c)
        intent = intents[i] if i < len(intents) else None

        if intent and action:
            if action in intent_to_tools.get(intent, []):
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
        else:
            rewards.append(0.0)

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
        elif "respond" in h:
            rewards.append(0.3)
        else:
            rewards.append(-0.3)
    return rewards


# ── TRL environment_factory reward adapter ───────────────────────────────────

def total_reward(environments, **kwargs):
    """
    Combined Reward function strictly scaled for PPO safety.
    Prioritizes Ground Truth Success > Efficiency > Format compliance.
    Fully clips reward to [-5.0, 5.0] to prevent explosion variance.

    When used with TRL environment_factory, `environments` is a list of
    SupportToolEnv instances. We extract state from each env to build
    the kwargs the sub-reward functions expect.
    """
    # Extract completions from kwargs (TRL passes these)
    completions = kwargs.get("completions", [""] * len(environments))

    # Build enriched kwargs from environment state
    enriched = dict(kwargs)
    enriched["grader_score"] = [env.reward for env in environments]
    enriched["final_reward"] = [env._cumulative_reward for env in environments]
    enriched["task_id"] = [env._task_id for env in environments]
    enriched["step_count"] = [env._env.state.step_count for env in environments]
    enriched["action_history"] = [list(env._env._tools_used) for env in environments]

    # Run all sub-reward functions with the original logic
    r1 = reward_task_success(completions, **enriched)
    r2 = reward_step_progress(completions, **enriched)
    r3 = reward_format(completions, **enriched)
    r4 = reward_conciseness(completions, **enriched)
    r5 = reward_repetition(completions, **enriched)
    r6 = reward_step_efficiency(completions, **enriched)
    r7 = reward_invalid_action(completions, **enriched)
    r8 = reward_completion(completions, **enriched)
    r9 = reward_action_alignment(completions, **enriched)

    r = []
    for i in range(len(environments)):
        total = (
            r1[i] * 2.0 +   # Success is paramount
            r2[i] * 1.0 +
            r3[i] * 0.5 +
            r4[i] * 0.5 +
            r5[i] * 0.7 +
            r6[i] * 0.8 +
            r7[i] * 1.5 +
            r8[i] * 1.0 +
            r9[i] * 1.2     # Action alignment weight
        )

        # PPO-safe clipping to prevent reward scale drift
        total = max(-5.0, min(5.0, total))
        r.append(total)

    return r
