"""
Supervisor Agent — OpenEnv CSA
==============================
Acts as the final quality control layer in the multi-agent pipeline. 
The supervisor reviews the specialist's work, addresses customer sentiment 
(especially frustration), and generates the final response or escalation action.
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
    The Supervisor reviews specialist output and generates the final response callback.
    It specializes in empathic communication and escalation management.
    """

    def __init__(self, client, model_id: str):
        """Initializes the supervisor with the API client and model identifier."""
        self.client = client
        self.model_id = model_id
        self.name = "Supervisor"
        self.emoji = "👨‍💼"

    def review_and_respond(self, customer_message: str, specialist_actions: List[str],
                           specialist_name: str, observation_text: str) -> str:
        """
        Processes specialist actions and environment feedback to produce the final turn action.
        Uses the OpenAI API to craft a helpful, polite response or escalate to human support.
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
        """
        Identifies high-urgency or negative sentiment signals that require 
        immediate attention from the Supervisor or a Human Manager.
        """
        escalation_signals = [
            "manager", "supervisor", "sue", "lawyer", "complaint",
            "pathetic", "terrible", "worst", "furious", "unacceptable"
        ]
        msg_lower = customer_message.lower()
        return any(signal in msg_lower for signal in escalation_signals)
