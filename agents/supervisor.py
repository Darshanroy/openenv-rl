"""
Supervisor Agent — Reviews specialist output, handles angry customers,
and makes the final decision: respond, request more info, or escalate.
Refactored to use the OpenAI API client for high-performance inference.
"""
from typing import List, Dict, Optional
import re

SUPERVISOR_SYSTEM_PROMPT = (
    "You are the SUPERVISOR Agent for customer support.\n"
    "Your job is to:\n"
    "1. Review the specialist agent's work and ensure the customer's issue is resolved.\n"
    "2. Handle angry or abusive customers with empathy.\n"
    "3. Escalate to a human manager when the issue is beyond automated resolution.\n\n"
    "Available tools: [escalate_to_human('reason')], [respond('message')]\n\n"
    "Rules:\n"
    "- If the customer is angry, always apologize first, then resolve.\n"
    "- If the specialist failed or the issue is unresolvable, escalate.\n"
    "- If everything looks good, generate the final customer-facing response.\n"
    "Format ALL tool calls in brackets WITH QUOTES around parameters: [tool_name('param')]\n"
    "Example: [respond('Your issue is resolved.')]"
)


class SupervisorAgent:
    """
    The Supervisor reviews specialist work and generates the final response via the OpenAI API client.
    For angry/escalation scenarios, it takes direct control.
    """

    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id
        self.name = "Supervisor"
        self.emoji = "👨‍💼"

    def review_and_respond(self, customer_message: str, specialist_actions: List[str],
                           specialist_name: str, observation_text: str) -> str:
        """
        Reviews what the specialist did and generates the final action via the OpenAI API client.
        """
        actions_summary = "\n".join([f"  - {a}" for a in specialist_actions]) if specialist_actions else "  (no actions taken)"

        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Customer said: \"{customer_message}\"\n\n"
                f"The {specialist_name} performed these actions:\n{actions_summary}\n\n"
                f"Latest environment feedback: {observation_text}\n\n"
                f"Your decision (respond to customer or escalate):"
            )}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=256
            )
            response = completion.choices[0].message.content or ""

            # Extract tool calls using strict regex
            valid_tools = "escalate_to_human|respond"
            tool_regex = rf'\[({valid_tools})\(.*?\)\]'
            match = re.search(tool_regex, response)
            if match:
                return match.group(0)

            return "[respond('I have reviewed the case and I am handing it over to you. Please let me know how I can help further.')]"
            
        except Exception as e:
            return f"[respond('I encountered an internal error. Please try again later.')]"

    def should_escalate(self, customer_message: str) -> bool:
        """Quick heuristic check if escalation is likely needed."""
        escalation_signals = [
            "manager", "supervisor", "sue", "lawyer", "complaint",
            "pathetic", "terrible", "worst", "furious", "unacceptable"
        ]
        msg_lower = customer_message.lower()
        return any(signal in msg_lower for signal in escalation_signals)
