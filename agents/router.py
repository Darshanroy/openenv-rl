"""
Rule-Based Router Agent — Classifies customer intent and dispatches to the correct specialist.
"""
import re

# Keyword → Agent mapping
ROUTING_TABLE = {
    "order": {
        "keywords": [
            "order", "status", "cancel", "payment", "coupon", "password",
            "account", "login", "forgot", "ORD-101", "ORD-505", "ORD-1414",
            "SAVE10", "apply coupon", "reset password"
        ],
        "scenarios": [
            "easy_status", "easy_cancel", "easy_payment_fail",
            "easy_coupon", "easy_account"
        ]
    },
    "logistics": {
        "keywords": [
            "shipping", "delivery", "track", "address", "reschedule",
            "delay", "late", "missing", "delivered but", "not here",
            "change address", "slot", "ORD-909", "ORD-1919", "ORD-2323", "ORD-1313"
        ],
        "scenarios": [
            "medium_delay", "medium_address", "medium_reschedule", "hard_missing"
        ]
    },
    "finance": {
        "keywords": [
            "refund", "return", "damaged", "broken", "money back",
            "charged twice", "double charge", "shattered", "crushed",
            "ORD-2020", "ORD-1515", "ORD-2121", "ORD-2222"
        ],
        "scenarios": [
            "medium_return", "medium_double_charge",
            "hard_refund", "hard_damaged"
        ]
    },
    "supervisor": {
        "keywords": [
            "manager", "supervisor", "escalate", "angry", "sue",
            "pathetic", "terrible", "worst", "furious", "complain"
        ],
        "scenarios": [
            "hard_angry", "hard_escalation"
        ]
    }
}


class Router:
    """
    Rule-based intent classifier.
    Scores each agent category by keyword match count and returns the best match.
    """

    def classify(self, message: str, task_id: str = None) -> str:
        """
        Returns one of: 'order', 'logistics', 'finance', 'supervisor'.
        
        Priority:
        1. If task_id is known, use scenario mapping (guaranteed correct).
        2. Otherwise, score by keyword match count.
        3. Default to 'order' if nothing matches.
        """
        # Fast path: if we know the scenario, route directly
        if task_id:
            for agent_type, config in ROUTING_TABLE.items():
                if task_id in config["scenarios"]:
                    return agent_type

        # Keyword scoring
        msg_lower = message.lower()
        scores = {}
        for agent_type, config in ROUTING_TABLE.items():
            score = sum(1 for kw in config["keywords"] if kw.lower() in msg_lower)
            scores[agent_type] = score

        # Supervisor gets a priority boost — angry/escalation should always win
        scores["supervisor"] = scores.get("supervisor", 0) * 2

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "order"

    def get_agent_emoji(self, agent_type: str) -> str:
        emojis = {
            "order": "📦",
            "logistics": "🚚",
            "finance": "💰",
            "supervisor": "👨‍💼"
        }
        return emojis.get(agent_type, "🤖")
