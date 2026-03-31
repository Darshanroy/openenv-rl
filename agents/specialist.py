"""
Specialist Agent — A reusable agent class that operates with a restricted tool set and
a focused system prompt. Each specialist only sees and uses its own tools.
"""
import torch
from typing import List, Dict, Optional


# Per-specialist configuration
SPECIALIST_CONFIGS = {
    "order": {
        "name": "Order Agent",
        "emoji": "📦",
        "allowed_tools": ["get_order", "get_order_status", "cancel_order", "validate_coupon", "reset_password"],
        "system_prompt": (
            "You are the ORDER SPECIALIST Agent.\n"
            "You handle: order lookups, cancellations, payment issues, coupon validation, and account recovery.\n"
            "Available tools: get_order(id), cancel_order(id), validate_coupon(code), reset_password(email).\n"
            "Format ALL tool calls in brackets WITH QUOTES around parameters: [tool_name('param')]\n"
            "Example: [get_order('ORD-123')]\n"
            "When done, use: [respond('your message to the customer')]\n"
            "Be concise and efficient. Solve in as few steps as possible."
        ),
    },
    "logistics": {
        "name": "Logistics Agent",
        "emoji": "🚚",
        "allowed_tools": ["get_order", "track_shipment", "update_address", "check_delivery_slot", "reschedule_delivery", "investigate_missing"],
        "system_prompt": (
            "You are the LOGISTICS SPECIALIST Agent.\n"
            "You handle: shipment tracking, delivery delays, address changes, rescheduling, and missing items.\n"
            "Available tools: get_order(id), track_shipment(id), update_address(id, addr), "
            "check_delivery_slot(id), reschedule_delivery(id, slot), investigate_missing(id).\n"
            "Format ALL tool calls in brackets WITH QUOTES around parameters: [tool_name('param')]\n"
            "Example: [track_shipment('ORD-123')]\n"
            "When done, use: [respond('your message to the customer')]\n"
            "Be concise and efficient."
        ),
    },
    "finance": {
        "name": "Finance Agent",
        "emoji": "💰",
        "allowed_tools": ["get_order", "validate_return", "ask_proof", "create_return_request", "initiate_refund", "get_payment_details"],
        "system_prompt": (
            "You are the FINANCE SPECIALIST Agent.\n"
            "You handle: returns, refunds, damage claims, and double charges.\n"
            "Available tools: get_order(id), validate_return(id), ask_proof(id), "
            "create_return_request(id), initiate_refund(id), get_payment_details(txn_id).\n"
            "Format ALL tool calls in brackets WITH QUOTES around parameters: [tool_name('param')]\n"
            "Example: [validate_return('ORD-123')]\n"
            "IMPORTANT: For damaged items, ALWAYS ask_proof() before initiating a refund.\n"
            "When done, use: [respond('your message to the customer')]\n"
            "Be concise and efficient."
        ),
    },
}


class SpecialistAgent:
    """
    A specialist agent that generates tool calls using its restricted system prompt.
    All specialists share the same underlying model but have different instructions.
    """

    def __init__(self, agent_type: str, model, tokenizer, device: str = "cpu"):
        if agent_type not in SPECIALIST_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}. Must be one of {list(SPECIALIST_CONFIGS.keys())}")

        self.agent_type = agent_type
        self.config = SPECIALIST_CONFIGS[agent_type]
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def emoji(self) -> str:
        return self.config["emoji"]

    @property
    def allowed_tools(self) -> List[str]:
        return self.config["allowed_tools"]

    def generate_action(self, observation_text: str, history_text: str = "") -> str:
        """
        Given the current observation (environment feedback), generate the next tool call.
        """
        prompt = (
            f"{self.config['system_prompt']}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Current observation: {observation_text}\n\n"
            f"Your next action:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the prompt)
        response = full_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

        # Regex extract the FIRST bracketed string to prevent over-generation
        import re
        match = re.search(r'\[.*?\]', response)
        if match:
            # If the model forgot quotes, gently fix it for the environment parser
            tool_call = match.group(0)
            # e.g. [get_order(ORD-101)] -> [get_order('ORD-101')]
            tool_call = re.sub(r'\((ORD-[0-9]+)\)', r"('\1')", tool_call)
            return tool_call
        
        return "[respond('I need more information to help you.')]"


    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a given tool is within this specialist's scope."""
        return tool_name in self.allowed_tools or tool_name == "respond"
