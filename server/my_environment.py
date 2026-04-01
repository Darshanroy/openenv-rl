"""
OpenEnv-compliant Customer Support Environment.
Inherits from openenv.core.Environment and implements reset(), step(), state property.
"""
import re
import ast
import random
import uuid
from typing import Dict, Any, Tuple, Optional, List

from openenv.core import Environment
from ..models import SupportAction, SupportObservation, SupportState
from .tools import ACTION_REGISTRY


# ── Task Configuration ──────────────────────────────────────────────────────────
# 15 tasks across 3 difficulty tiers (Easy / Medium / Hard)

TASK_CONFIGS = {
    "easy_status": {
        "difficulty": "easy",
        "customer": "Darshan Sharma",
        "initial_message": "Where is my Boat headset? ORD-101",
        "target_tools": ["get_order", "track_shipment"],
        "intent_rewards": {"get_order": 5, "track_shipment": 5, "respond": 5},
        "optimal_steps": 3,
        "grader_weights": {"get_order": 0.4, "track_shipment": 0.5, "respond": 0.1},
    },
    "medium_delay": {
        "difficulty": "medium",
        "customer": "Vikram Singh",
        "initial_message": "My MacBook (ORD-909) is a week late! What's the hold up?",
        "target_tools": ["get_order", "track_shipment"],
        "intent_rewards": {"get_order": 4, "track_shipment": 4, "respond": 4},
        "optimal_steps": 3,
        "grader_weights": {"get_order": 0.2, "track_shipment": 0.6, "respond": 0.2},
    },
    "easy_cancel": {
        "difficulty": "easy",
        "customer": "Priya Sharma",
        "initial_message": "Cancel my order ORD-505 immediately.",
        "target_tools": ["get_order", "cancel_order"],
        "intent_rewards": {"get_order": 3, "cancel_order": 8},
        "optimal_steps": 2,
        "grader_weights": {"get_order": 0.3, "cancel_order": 0.6, "respond": 0.1},
    },
    "medium_return": {
        "difficulty": "medium",
        "customer": "Fatima Saeed",
        "initial_message": "I want to return my MK handbag (ORD-2020).",
        "target_tools": ["get_order", "validate_return", "create_return_request"],
        "intent_rewards": {"validate_return": 3, "create_return_request": 8},
        "optimal_steps": 3,
        "grader_weights": {"validate_return": 0.3, "create_return_request": 0.6, "respond": 0.1},
    },
    "hard_refund": {
        "difficulty": "hard",
        "customer": "Chris Evans",
        "initial_message": "My order ORD-2121 was cancelled but I didn't get my money back! Refund me now.",
        "target_tools": ["get_order", "validate_return", "initiate_refund"],
        "intent_rewards": {"validate_return": 3, "initiate_refund": 10},
        "optimal_steps": 3,
        "grader_weights": {"validate_return": 0.3, "initiate_refund": 0.6, "respond": 0.1},
    },
    "hard_damaged": {
        "difficulty": "hard",
        "customer": "Pooja Hegde",
        "initial_message": "My iPhone 15 (ORD-2222) back is shattered!! I want a refund.",
        "target_tools": ["get_order", "ask_proof", "validate_return", "initiate_refund"],
        "intent_rewards": {"ask_proof": 2, "validate_return": 3, "initiate_refund": 10},
        "optimal_steps": 4,
        "grader_weights": {"ask_proof": 0.2, "initiate_refund": 0.7, "respond": 0.1},
    },
    "medium_address": {
        "difficulty": "medium",
        "customer": "Arjun Desai",
        "initial_message": "Can I change my delivery address for ORD-1919?",
        "target_tools": ["get_order", "update_address"],
        "intent_rewards": {"get_order": 3, "update_address": 6},
        "optimal_steps": 2,
        "grader_weights": {"update_address": 0.8, "respond": 0.2},
    },
    "medium_reschedule": {
        "difficulty": "medium",
        "customer": "Rajesh Kumar",
        "initial_message": "I'm not home for ORD-2323. Can you deliver it later?",
        "target_tools": ["get_order", "check_delivery_slot", "reschedule_delivery"],
        "intent_rewards": {"check_delivery_slot": 2, "reschedule_delivery": 6},
        "optimal_steps": 3,
        "grader_weights": {"reschedule_delivery": 0.8, "respond": 0.2},
    },
    "hard_missing": {
        "difficulty": "hard",
        "customer": "Kabir Khan",
        "initial_message": "My ORD-1313 says delivered but it's not here!",
        "target_tools": ["get_order", "track_shipment", "investigate_missing", "escalate_to_human"],
        "intent_rewards": {"investigate_missing": 5, "escalate_to_human": 5},
        "optimal_steps": 4,
        "grader_weights": {"investigate_missing": 0.5, "escalate_to_human": 0.4, "respond": 0.1},
    },
    "easy_payment_fail": {
        "difficulty": "easy",
        "customer": "Sofia Rossi",
        "initial_message": "My payment for ORD-1414 failed but money was deducted.",
        "target_tools": ["get_order", "respond"],
        "intent_rewards": {"get_order": 5, "respond": 5},
        "optimal_steps": 2,
        "grader_weights": {"get_order": 0.5, "respond": 0.5},
    },
    "medium_double_charge": {
        "difficulty": "medium",
        "customer": "Rajesh Kumar",
        "initial_message": "I was charged twice for my gas hob order ORD-1515!",
        "target_tools": ["get_order", "initiate_refund"],
        "intent_rewards": {"initiate_refund": 12},
        "optimal_steps": 2,
        "grader_weights": {"initiate_refund": 0.9, "respond": 0.1},
    },
    "easy_coupon": {
        "difficulty": "easy",
        "customer": "Amit Patel",
        "initial_message": "My coupon SAVE10 isn't working for my order.",
        "target_tools": ["validate_coupon", "respond"],
        "intent_rewards": {"validate_coupon": 4, "respond": 2},
        "optimal_steps": 2,
        "grader_weights": {"validate_coupon": 0.8, "respond": 0.2},
    },
    "easy_account": {
        "difficulty": "easy",
        "customer": "Meera Reddy",
        "initial_message": "I forgot my password for meera.reddy@example.com.",
        "target_tools": ["reset_password", "respond"],
        "intent_rewards": {"reset_password": 5, "respond": 2},
        "optimal_steps": 2,
        "grader_weights": {"reset_password": 0.9, "respond": 0.1},
    },
    "hard_angry": {
        "difficulty": "hard",
        "customer": "Vikram Singh",
        "initial_message": "YOUR SERVICE IS PATHETIC! FIX MY ORDER ORD-909 NOW OR I SUE!",
        "target_tools": ["get_order", "track_shipment", "respond"],
        "intent_rewards": {"respond": 5, "track_shipment": 3},
        "optimal_steps": 3,
        "grader_weights": {"respond": 0.5, "track_shipment": 0.5},
    },
    "hard_escalation": {
        "difficulty": "hard",
        "customer": "Sofia Rossi",
        "initial_message": "I want to talk to your manager about ORD-1414. Now.",
        "target_tools": ["get_order", "escalate_to_human"],
        "intent_rewards": {"escalate_to_human": 5},
        "optimal_steps": 2,
        "grader_weights": {"escalate_to_human": 0.9, "respond": 0.1},
    },
}


class SupportEnvironment(Environment):
    """
    OpenEnv-compliant Customer Support Environment.

    Implements the full Environment interface:
      - reset(seed, episode_id, **kwargs) -> SupportObservation
      - step(action: SupportAction, **kwargs) -> SupportObservation
      - state (property) -> SupportState
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        # Per-episode state
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._task_id: str = ""
        self._scenario: dict = {}
        self._tools_used: set = set()
        self._messages: List[Message] = []
        self._prompt: str = ""
        self._done: bool = False
        self._variables: Dict[str, Any] = {}

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> SupportObservation:
        """Start a new customer support episode."""
        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id")
        if task_id and task_id in TASK_CONFIGS:
            task = TASK_CONFIGS[task_id]
        else:
            task_id = random.choice(list(TASK_CONFIGS.keys()))
            task = TASK_CONFIGS[task_id]

        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._step_count = 0
        self._task_id = task_id
        self._scenario = task
        self._tools_used = set()
        self._done = False
        self._variables = {"difficulty": task["difficulty"]}

        self._prompt = (
            f"SYSTEM: You are an E-Commerce Customer Support Agent. Task: {task_id}\n"
            "Format tool calls in brackets: [tool_name('param1', 'param2')]\n"
            "Available tools: get_order, track_shipment, validate_return, initiate_refund, "
            "cancel_order, update_address, check_delivery_slot, reschedule_delivery, "
            "investigate_missing, ask_proof, validate_coupon, reset_password, "
            "escalate_to_human, respond."
        )

        self._messages = [
            {"role": "customer", "content": task["initial_message"]}
        ]

        return SupportObservation(
            prompt=self._prompt,
            messages=list(self._messages),
            done=False,
            reward=0.0,
        )

    def step(self, action: SupportAction, timeout_s: Optional[float] = None, **kwargs) -> SupportObservation:
        """Process one agent action and return the next observation."""
        if self._done:
            return SupportObservation(
                prompt=self._prompt,
                messages=list(self._messages),
                done=True,
                reward=0.0,
                metadata={"grader_score": self._calculate_grader_score()},
            )

        self._step_count += 1
        guess = action.message.strip()
        self._messages.append({"role": "agent", "content": guess})

        func_name, args, kwargs_parsed, error_msg = self._parse_action_string(guess)
        feedback = ""
        reward = -1.0  # Global step penalty

        if error_msg:
            feedback = f"Syntax Error: {error_msg}"
            reward -= 4.0
        elif func_name == "respond":
            feedback = "Task concluded by Agent response."
            self._tools_used.add("respond")
            reward += self._scenario.get("intent_rewards", {}).get("respond", 0)
            self._done = True
            reward += 10.0  # Resolution bonus
        elif func_name in ACTION_REGISTRY:
            try:
                res = ACTION_REGISTRY[func_name](*args, **kwargs_parsed)
                feedback = f"API Output: {str(res)}"
                self._tools_used.add(func_name)
                reward += self._scenario.get("intent_rewards", {}).get(func_name, 0)
                if func_name == "escalate_to_human":
                    self._done = True
                    reward += 10.0
                    feedback += " Task Escalated."
            except Exception as e:
                feedback = f"Execution Error: {e}"
                reward -= 5.0
        else:
            feedback = f"Unknown Tool '{func_name}'"
            reward -= 5.0

        self._messages.append({"role": "system", "content": feedback})

        if self._step_count >= self.max_turns:
            self._done = True

        # Efficiency bonus
        if self._done and self._step_count <= self._scenario.get("optimal_steps", 99):
            reward += 3.0

        info = {}
        if self._done:
            info["grader_score"] = self._calculate_grader_score()
            info["task_id"] = self._task_id

        return SupportObservation(
            prompt=self._prompt,
            messages=list(self._messages),
            done=self._done,
            reward=float(reward),
            metadata=info,
        )

    @property
    def state(self) -> SupportState:
        """Return current episode state (OpenEnv spec)."""
        return SupportState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            max_steps=self.max_turns,
            tools_used=list(self._tools_used),
            variables=self._variables,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_action_string(self, action_str: str) -> Tuple[Optional[str], list, dict, str]:
        match = re.search(r'\[(.*?)\]', action_str, re.DOTALL)
        if not match:
            return None, [], {}, "No bracketed tool call found."
        inner = match.group(1).strip()

        if inner.startswith("respond(") or inner.startswith("respond "):
            text = inner.replace("respond(", "").replace("respond ", "").strip("')\"\\ ")
            return "respond", [text], {}, ""
        try:
            tree = ast.parse(inner, mode='eval')
            if isinstance(tree.body, ast.Call):
                return (
                    tree.body.func.id,
                    [ast.literal_eval(arg) for arg in tree.body.args],
                    {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords},
                    "",
                )
            return None, [], {}, "Format: [func(args)]"
        except Exception as e:
            return None, [], {}, str(e)

    def _calculate_grader_score(self) -> float:
        """
        Calculate weighted score (0.0 to 1.0) with PROTOCOL ENFORCEMENT.
        Conditional triggers ensure agents don't 'guess' their way to a solve.
        """
        weights = self._scenario.get("grader_weights", {})
        used = self._tools_used
        score = 0.0

        # Protocols: Define required tool sequences
        protocols = {
            "hard_damaged": ("ask_proof", "initiate_refund"),
            "hard_missing": ("investigate_missing", "escalate_to_human"),
            "medium_return": ("validate_return", "create_return_request"),
            "hard_refund": ("validate_return", "initiate_refund")
        }

        # Check for blocked tools
        blocked_tools = set()
        if self._task_id in protocols:
            prereq, main_tool = protocols[self._task_id]
            if main_tool in used and prereq not in used:
                blocked_tools.add(main_tool)

        # sum weights for allowed tools
        for tool, weight in weights.items():
            if tool in used and tool not in blocked_tools:
                score += weight
        
        return min(float(score), 1.0)
