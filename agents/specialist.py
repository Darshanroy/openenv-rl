"""
Specialist Agent — A reusable agent class that operates with a restricted tool set and
a focused system prompt. Each specialist only sees and uses its own tools.
Refactored to use the OpenAI API client for high-performance inference.
"""
from typing import List, Dict, Optional, Tuple
import re

# Per-specialist configuration
SPECIALIST_CONFIGS = {
    "order": {
        "name": "Order Agent",
        "emoji": "📦",
        "allowed_tools": ["get_order", "get_order_status", "cancel_order", "validate_coupon", "reset_password"],
        "system_prompt": (
            "You are the ORDER SPECIALIST Agent. You handle: order lookups, status checks, cancellations, and coupons.\n"
            "Available tools: get_order(id), cancel_order(id), validate_coupon(code), reset_password(email).\n"
            "Format your response exactly as follows:\n"
            "<thought>\nYour internal reasoning (background on what you found and next steps)\n</thought>\n"
            "[tool_name('parameter')]\n"
            "Example: [get_order('ORD-101')]\n"
            "When done, use [respond('A friendly message for the customer.')]\n"
            "IMPORTANT: Replace 'tool_name' with the actual tool and 'parameter' with the actual value (e.g. ORD-2222)."
        ),
    },
    "logistics": {
        "name": "Logistics Agent",
        "emoji": "🚚",
        "allowed_tools": ["get_order", "track_shipment", "update_address", "check_delivery_slot", "reschedule_delivery", "investigate_missing"],
        "system_prompt": (
            "You are the LOGISTICS SPECIALIST Agent.\n"
            "You handle: shipment tracking, delivery delays, and address changes.\n"
            "Available tools: get_order(id), track_shipment(id), update_address(id, addr), check_delivery_slot(id), reschedule_delivery(id, slot), investigate_missing(id).\n"
            "Format your response as follows:\n"
            "<thought>\nYour internal reasoning\n</thought>\n"
            "[tool_name('parameter')]\n"
            "Example: [track_shipment('ORD-101')]\n"
            "When done, use [respond('A friendly message for the customer.')]\n"
            "IMPORTANT: Replace 'tool_name' with the actual tool and 'parameter' with the actual value."
        ),
    },
    "finance": {
        "name": "Finance Agent",
        "emoji": "💰",
        "allowed_tools": ["get_order", "validate_return", "ask_proof", "create_return_request", "initiate_refund", "get_payment_details"],
        "system_prompt": (
            "You are the FINANCE SPECIALIST Agent.\n"
            "You handle: returns, refunds, and damage claims.\n"
            "Available tools: get_order(id), validate_return(id), ask_proof(id), create_return_request(id), initiate_refund(id), get_payment_details(txn_id).\n"
            "Format your response as follows:\n"
            "<thought>\nYour internal reasoning\n</thought>\n"
            "[tool_name('parameter')]\n"
            "Example: [validate_return('ORD-101')]\n"
            "When done, use [respond('A friendly message for the customer.')]\n"
            "IMPORTANT: Replace 'tool_name' with the actual tool and 'parameter' with the actual value."
        ),
    },
}


class SpecialistAgent:
    """
    A specialist agent that generates tool calls using its restricted system prompt.
    All specialists share the same underlying model API but have different instructions.
    """

    def __init__(self, agent_type: str, client, model_id: str):
        if agent_type not in SPECIALIST_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}. Must be one of {list(SPECIALIST_CONFIGS.keys())}")

        self.agent_type = agent_type
        self.config = SPECIALIST_CONFIGS[agent_type]
        self.client = client
        self.model_id = model_id

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def emoji(self) -> str:
        return self.config["emoji"]

    @property
    def allowed_tools(self) -> List[str]:
        return self.config["allowed_tools"]

    def generate_action(self, observation_text: str, history_text: str = "") -> Tuple[str, str]:
        """
        Given the current observation (environment feedback), generate thoughts and the next tool call via the OpenAI API client.
        """
        # History window (approx. 5-7 turns) for end-to-end resolution.
        history_lines = history_text.split("\n")
        short_history = "\n".join(history_lines[-8:])

        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": f"History:\n{short_history}\n\nObs: {observation_text}\n\nDecision:"}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.2, # Lower temperature for tool calling precision
                max_tokens=256
            )
            response = completion.choices[0].message.content or ""

            # Parse Thought and Action
            thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else "Analyzing current observation..."
            
            tool_match = re.search(r'\[.*?\]', response)
            if tool_match:
                action = tool_match.group(0)
                # Gentle cleanup to ensure valid action format (remove unnecessary parentheses or spaces)
                action = re.sub(r'\((ORD-[0-9]+)\)', r"('\1')", action)
                return thought, action
            
            # If no tool found, default to respondent
            return thought, f"[respond('I need more information to assist you regarding {self.agent_type}.')]"

        except Exception as e:
            return f"Error in LLM: {str(e)}", "[respond('I encountered an internal error. Please wait.')]"


    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a given tool is within this specialist's scope."""
        return tool_name in self.allowed_tools or tool_name == "respond"
