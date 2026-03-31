import re
import ast
import random
from typing import Dict, Any, Tuple, Optional, List
from ..models import Observation, Action, EnvResult, Message, State
from .tools import ACTION_REGISTRY

# Task Configuration
# Easy: Order Status
# Medium: Return/Refund
# Hard: Escalated Fraud/Security

TASK_CONFIGS = {
    "easy_status": {
        "difficulty": "easy",
        "customer": "Darshan Sharma",
        "initial_message": "Where is my Boat headset? ORD-101",
        "target_tools": ["get_order", "track_shipment"],
        "intent_rewards": {"get_order": 5, "track_shipment": 5, "respond": 5},
        "optimal_steps": 3,
        "grader_weights": {"get_order": 0.4, "track_shipment": 0.5, "respond": 0.1}
    },
    "medium_delay": {
        "difficulty": "medium",
        "customer": "Vikram Singh",
        "initial_message": "My MacBook (ORD-909) is a week late! What's the hold up?",
        "target_tools": ["get_order", "track_shipment"],
        "intent_rewards": {"get_order": 4, "track_shipment": 4, "respond": 4},
        "optimal_steps": 3,
        "grader_weights": {"get_order": 0.2, "track_shipment": 0.6, "respond": 0.2}
    },
    "easy_cancel": {
        "difficulty": "easy",
        "customer": "Priya Sharma",
        "initial_message": "Cancel my order ORD-505 immediately.",
        "target_tools": ["get_order", "cancel_order"],
        "intent_rewards": {"get_order": 3, "cancel_order": 8},
        "optimal_steps": 2,
        "grader_weights": {"get_order": 0.3, "cancel_order": 0.6, "respond": 0.1}
    },
    "medium_return": {
        "difficulty": "medium",
        "customer": "Fatima Saeed",
        "initial_message": "I want to return my MK handbag (ORD-2020).",
        "target_tools": ["get_order", "validate_return", "create_return_request"],
        "intent_rewards": {"validate_return": 3, "create_return_request": 8},
        "optimal_steps": 3,
        "grader_weights": {"validate_return": 0.3, "create_return_request": 0.6, "respond": 0.1}
    },
    "hard_refund": {
        "difficulty": "hard",
        "customer": "Chris Evans",
        "initial_message": "My order ORD-2121 was cancelled but I didn't get my money back! Refund me now.",
        "target_tools": ["get_order", "validate_return", "initiate_refund"],
        "intent_rewards": {"validate_return": 3, "initiate_refund": 10},
        "optimal_steps": 3,
        "grader_weights": {"validate_return": 0.3, "initiate_refund": 0.6, "respond": 0.1}
    },
    "hard_damaged": {
        "difficulty": "hard",
        "customer": "Pooja Hegde",
        "initial_message": "My iPhone 15 (ORD-2222) back is shattered!! I want a refund.",
        "target_tools": ["get_order", "ask_proof", "validate_return", "initiate_refund"],
        "intent_rewards": {"ask_proof": 2, "validate_return": 3, "initiate_refund": 10},
        "optimal_steps": 4,
        "grader_weights": {"ask_proof": 0.2, "initiate_refund": 0.7, "respond": 0.1}
    },
    "medium_address": {
        "difficulty": "medium",
        "customer": "Arjun Desai",
        "initial_message": "Can I change my delivery address for ORD-1919?",
        "target_tools": ["get_order", "update_address"],
        "intent_rewards": {"get_order": 3, "update_address": 6},
        "optimal_steps": 2,
        "grader_weights": {"update_address": 0.8, "respond": 0.2}
    },
    "medium_reschedule": {
        "difficulty": "medium",
        "customer": "Rajesh Kumar",
        "initial_message": "I'm not home for ORD-2323. Can you deliver it later?",
        "target_tools": ["get_order", "check_delivery_slot", "reschedule_delivery"],
        "intent_rewards": {"check_delivery_slot": 2, "reschedule_delivery": 6},
        "optimal_steps": 3,
        "grader_weights": {"reschedule_delivery": 0.8, "respond": 0.2}
    },
    "hard_missing": {
        "difficulty": "hard",
        "customer": "Kabir Khan",
        "initial_message": "My ORD-1313 says delivered but it's not here!",
        "target_tools": ["get_order", "track_shipment", "investigate_missing", "escalate_to_human"],
        "intent_rewards": {"investigate_missing": 5, "escalate_to_human": 5},
        "optimal_steps": 4,
        "grader_weights": {"investigate_missing": 0.5, "escalate_to_human": 0.4, "respond": 0.1}
    },
    "easy_payment_fail": {
        "difficulty": "easy",
        "customer": "Sofia Rossi",
        "initial_message": "My payment for ORD-1414 failed but money was deducted.",
        "target_tools": ["get_order", "respond"],
        "intent_rewards": {"get_order": 5, "respond": 5},
        "optimal_steps": 2,
        "grader_weights": {"get_order": 0.5, "respond": 0.5}
    },
    "medium_double_charge": {
        "difficulty": "medium",
        "customer": "Rajesh Kumar",
        "initial_message": "I was charged twice for my gas hob order ORD-1515!",
        "target_tools": ["get_order", "initiate_refund"],
        "intent_rewards": {"initiate_refund": 12},
        "optimal_steps": 2,
        "grader_weights": {"initiate_refund": 0.9, "respond": 0.1}
    },
    "easy_coupon": {
        "difficulty": "easy",
        "customer": "Amit Patel",
        "initial_message": "My coupon SAVE10 isn't working for my order.",
        "target_tools": ["validate_coupon", "respond"],
        "intent_rewards": {"validate_coupon": 4, "respond": 2},
        "optimal_steps": 2,
        "grader_weights": {"validate_coupon": 0.8, "respond": 0.2}
    },
    "easy_account": {
        "difficulty": "easy",
        "customer": "Meera Reddy",
        "initial_message": "I forgot my password for meera.reddy@example.com.",
        "target_tools": ["reset_password", "respond"],
        "intent_rewards": {"reset_password": 5, "respond": 2},
        "optimal_steps": 2,
        "grader_weights": {"reset_password": 0.9, "respond": 0.1}
    },
    "hard_angry": {
        "difficulty": "hard",
        "customer": "Vikram Singh",
        "initial_message": "YOUR SERVICE IS PATHETIC! FIX MY ORDER ORD-909 NOW OR I SUE!",
        "target_tools": ["get_order", "track_shipment", "respond"],
        "intent_rewards": {"respond": 5, "track_shipment": 3},
        "optimal_steps": 3,
        "grader_weights": {"respond": 0.5, "track_shipment": 0.5}
    },
    "hard_escalation": {
        "difficulty": "hard",
        "customer": "Sofia Rossi",
        "initial_message": "I want to talk to your manager about ORD-1414. Now.",
        "target_tools": ["get_order", "escalate_to_human"],
        "intent_rewards": {"escalate_to_human": 5},
        "optimal_steps": 2,
        "grader_weights": {"escalate_to_human": 0.9, "respond": 0.1}
    }
}

class SupportEnvironment:
    """
    Core Realistic Sandbox Logic compliant with Full OpenEnv Spec.
    """
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.sessions: Dict[str, dict] = {} 

    def state(self, session_id: str) -> State:
        if session_id not in self.sessions:
             return State(session_id=session_id, current_turn=0, max_turns=self.max_turns, done=True)
        s = self.sessions[session_id]
        return State(
            session_id=session_id,
            current_turn=s["current_turn"],
            max_turns=self.max_turns,
            history=list(s["messages"]),
            tools_used=list(s["tools_used"]),
            variables=s.get("variables", {}),
            done=s["done"]
        )

    def reset(self, session_id: str = "default_session", task_id: Optional[str] = None) -> EnvResult:
        if task_id and task_id in TASK_CONFIGS:
            task = TASK_CONFIGS[task_id]
        else:
            task_id = random.choice(list(TASK_CONFIGS.keys()))
            task = TASK_CONFIGS[task_id]
        
        base_prompt = (
            f"SYSTEM: You are an E-Commerce Agent. Task: {task_id}\n"
            "Format tool calls in brackets: [tool_name(args)]\n"
            "Tools: get_order, track_shipment, validate_return, initiate_refund, cancel_order, "
            "update_address, check_delivery_slot, reschedule_delivery, investigate_missing, "
            "ask_proof, validate_coupon, reset_password, escalate_to_human, respond."
        )
        
        self.sessions[session_id] = {
            "task_id": task_id,
            "current_turn": 0,
            "scenario": task,
            "tools_used": set(),
            "messages": [Message(category="CUSTOMER", content=task["initial_message"])],
            "prompt": base_prompt,
            "done": False,
            "variables": {"difficulty": task["difficulty"]}
        }
        
        return EnvResult(
            observation=Observation(prompt=base_prompt, messages=list(self.sessions[session_id]["messages"])),
            reward=0.0,
            done=False,
            info={"task_id": task_id}
        )

    def _parse_action_string(self, action_str: str) -> Tuple[Optional[str], list, dict, str]:
        match = re.search(r'\[(.*?)\]', action_str, re.DOTALL)
        if not match: return None, [], {}, "No bracketed tool call found."
        inner = match.group(1).strip()
        if inner.startswith("respond(") or inner.startswith("respond "):
            text = inner.replace("respond(", "").replace("respond ", "").strip("')\" ")
            return "respond", [text], {}, ""
        try:
            tree = ast.parse(inner, mode='eval')
            if isinstance(tree.body, ast.Call):
                return tree.body.func.id, [ast.literal_eval(arg) for arg in tree.body.args], {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}, ""
            return None, [], {}, "Format: [func(args)]"
        except Exception as e: return None, [], {}, str(e)

    def _calculate_grader_score(self, session_id: str) -> float:
        s = self.sessions[session_id]
        weights = s["scenario"]["grader_weights"]
        score = 0.0
        used = s["tools_used"]
        for tool, weight in weights.items():
            if tool in used: score += weight
        return min(float(score), 1.0)

    def step(self, action: Action, session_id: str = "default_session") -> EnvResult:
        if session_id not in self.sessions: return self.reset(session_id)
        s = self.sessions[session_id]
        if s["done"]: return EnvResult(observation=Observation(prompt="", messages=s["messages"]), done=True)
        
        s["current_turn"] += 1
        guess = action.message.strip()
        s["messages"].append(Message(category="AGENT", content=guess))
        
        func_name, args, kwargs, error_msg = self._parse_action_string(guess)
        feedback = ""
        reward = -1.0 # Global step penalty

        if error_msg:
            feedback = f"Syntax Error: {error_msg}"; reward -= 4.0
        elif func_name == "respond":
            feedback = "Task concluded by Agent response."; s["tools_used"].add("respond")
            reward += s["scenario"]["intent_rewards"].get("respond", 0)
            s["done"] = True; reward += 10.0 # Resolution bonus
        elif func_name in ACTION_REGISTRY:
            try:
                res = ACTION_REGISTRY[func_name](*args, **kwargs)
                feedback = f"API Output: {str(res)}"; s["tools_used"].add(func_name)
                print(f"DEBUG: Session {session_id} used {func_name}. Current tools: {s['tools_used']}")
                # Map Master Table Rewards
                reward += s["scenario"]["intent_rewards"].get(func_name, 0)
                if func_name == "escalate_to_human": 
                    s["done"] = True; reward += 10.0; feedback += " Task Escalated."
            except Exception as e:
                feedback = f"Execution Error: {e}"; reward -= 5.0
        else:
            feedback = f"Unknown Tool '{func_name}'"; reward -= 5.0

        s["messages"].append(Message(category="FEEDBACK", content=feedback))
        if s["current_turn"] >= self.max_turns: s["done"] = True
        
        # Grading
        info = {"grader_score": self._calculate_grader_score(session_id)} if s["done"] else {}
        if s["done"] and s["current_turn"] <= s["scenario"].get("optimal_steps", 99): reward += 3.0 # Efficiency bonus

        return EnvResult(
            observation=Observation(prompt=s["prompt"], messages=list(s["messages"])),
            reward=float(reward),
            done=s["done"],
            info=info
        )
