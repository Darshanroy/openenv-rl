"""
Supervisor Agent — Reviews specialist output, handles angry customers,
and makes the final decision: respond, request more info, or escalate.
"""
import torch
from typing import List, Dict, Optional


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
    "Format ALL tool calls in brackets: [tool_name('param')]"
)


class SupervisorAgent:
    """
    The Supervisor reviews specialist work and generates the final response.
    For angry/escalation scenarios, it takes direct control.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.name = "Supervisor"
        self.emoji = "👨‍💼"

    def review_and_respond(self, customer_message: str, specialist_actions: List[str],
                           specialist_name: str, observation_text: str) -> str:
        """
        Reviews what the specialist did and generates the final action.
        """
        actions_summary = "\n".join([f"  - {a}" for a in specialist_actions]) if specialist_actions else "  (no actions taken)"

        prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"Customer said: \"{customer_message}\"\n\n"
            f"The {specialist_name} performed these actions:\n{actions_summary}\n\n"
            f"Latest environment feedback: {observation_text}\n\n"
            f"Your decision (respond to customer or escalate):"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

        return response if response else "[respond('Your issue has been resolved. Thank you for your patience.')]"

    def should_escalate(self, customer_message: str) -> bool:
        """Quick heuristic check if escalation is likely needed."""
        escalation_signals = [
            "manager", "supervisor", "sue", "lawyer", "complaint",
            "pathetic", "terrible", "worst", "furious", "unacceptable"
        ]
        msg_lower = customer_message.lower()
        return any(signal in msg_lower for signal in escalation_signals)
