"""
TRL-native environment_factory class for the Customer Support Environment.

This replaces the old rollout_func + REST client pattern with TRL's built-in
multi-turn tool-calling loop. Each public method (other than reset) is
automatically discovered by GRPOTrainer as a callable tool.

Reference: https://huggingface.co/docs/trl/openenv
"""
import sys
import os
import random

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env.server.my_environment import SupportEnvironment, TASK_CONFIGS
from my_env.models import SupportAction
from my_env.server.tools import ACTION_REGISTRY


class SupportToolEnv:
    """
    TRL environment_factory class for Customer Support training.

    GRPOTrainer calls:
      1. __init__()            — per generation
      2. reset(**kwargs)       — start episode, returns initial observation string
      3. <tool methods>        — called when model generates tool calls
      4. reward_func(environments) reads env.reward after episode

    All public methods (except reset) are exposed as tools the model can call.
    """

    def __init__(self):
        self._env = SupportEnvironment(max_turns=8)
        self.reward = 0.0
        self.done = False
        self._cumulative_reward = 0.0
        self._task_id = ""

    def reset(self, **kwargs) -> str | None:
        """
        Start a new customer support episode.
        Returns the initial observation as a string for the model.
        """
        self.reward = 0.0
        self.done = False
        self._cumulative_reward = 0.0

        # Pick a random task
        self._task_id = random.choice(list(TASK_CONFIGS.keys()))
        obs = self._env.reset(task_id=self._task_id)

        # Build initial context string for the model
        customer_msg = obs.messages[0]["content"] if obs.messages else ""
        return (
            f"Customer Support Task: {self._task_id}\n"
            f"Customer says: \"{customer_msg}\"\n\n"
            f"You have the following tools available. Use them to resolve the customer's issue.\n"
            f"When you are done, call `respond` with your final message to the customer."
        )

    # ── Tool Methods (auto-discovered by TRL) ────────────────────────────────

    def get_order(self, order_id: str) -> str:
        """
        Look up order details by order ID.

        Args:
            order_id: The order identifier, e.g. 'ORD-101'

        Returns:
            Order details including status, items, and delivery date.
        """
        return self._execute_tool("get_order", order_id)

    def track_shipment(self, order_id: str) -> str:
        """
        Track the shipment status for an order.

        Args:
            order_id: The order identifier to track

        Returns:
            Shipment location, courier, and estimated arrival.
        """
        return self._execute_tool("track_shipment", order_id)

    def cancel_order(self, order_id: str) -> str:
        """
        Cancel a pending order.

        Args:
            order_id: The order identifier to cancel

        Returns:
            Confirmation or rejection of the cancellation.
        """
        return self._execute_tool("cancel_order", order_id)

    def validate_return(self, order_id: str) -> str:
        """
        Check if an order is eligible for return.

        Args:
            order_id: The order identifier to validate

        Returns:
            Whether the return is valid and the reason.
        """
        return self._execute_tool("validate_return", order_id)

    def create_return_request(self, order_id: str) -> str:
        """
        Create a return request for a delivered order.

        Args:
            order_id: The order identifier to return

        Returns:
            Confirmation of the return request.
        """
        return self._execute_tool("create_return_request", order_id)

    def initiate_refund(self, order_id: str) -> str:
        """
        Initiate a refund for an order.

        Args:
            order_id: The order identifier to refund

        Returns:
            Refund confirmation or failure message.
        """
        return self._execute_tool("initiate_refund", order_id)

    def update_address(self, order_id: str, new_address: str) -> str:
        """
        Update the delivery address for an order.

        Args:
            order_id: The order identifier
            new_address: The new delivery address

        Returns:
            Confirmation of the address update.
        """
        return self._execute_tool("update_address", order_id, new_address)

    def check_delivery_slot(self, order_id: str) -> str:
        """
        Check available delivery slots for rescheduling.

        Args:
            order_id: The order identifier

        Returns:
            List of available delivery time slots.
        """
        return self._execute_tool("check_delivery_slot", order_id)

    def reschedule_delivery(self, order_id: str, slot: str) -> str:
        """
        Reschedule delivery to a new time slot.

        Args:
            order_id: The order identifier
            slot: The desired delivery time slot

        Returns:
            Confirmation of the rescheduled delivery.
        """
        return self._execute_tool("reschedule_delivery", order_id, slot)

    def investigate_missing(self, order_id: str) -> str:
        """
        Open an investigation for a missing package.

        Args:
            order_id: The order identifier reported missing

        Returns:
            Investigation case ID and next steps.
        """
        return self._execute_tool("investigate_missing", order_id)

    def ask_proof(self, order_id: str) -> str:
        """
        Request photo proof of damage from the customer.

        Args:
            order_id: The order identifier with reported damage

        Returns:
            Confirmation that a proof request was sent.
        """
        return self._execute_tool("ask_proof", order_id)

    def validate_coupon(self, code: str) -> str:
        """
        Validate a coupon code.

        Args:
            code: The coupon code to validate, e.g. 'SAVE10'

        Returns:
            Whether the coupon is valid and the discount amount.
        """
        return self._execute_tool("validate_coupon", code)

    def reset_password(self, email: str) -> str:
        """
        Send a password reset link to the customer's email.

        Args:
            email: The customer's email address

        Returns:
            Confirmation that the reset link was sent.
        """
        return self._execute_tool("reset_password", email)

    def escalate_to_human(self, issue: str) -> str:
        """
        Escalate the issue to a human support agent.

        Args:
            issue: Brief description of the issue to escalate

        Returns:
            Confirmation that the issue has been escalated.
        """
        return self._execute_tool("escalate_to_human", issue)

    def respond(self, message: str) -> str:
        """
        Send a final response message to the customer and end the episode.

        Args:
            message: The response message to send to the customer

        Returns:
            Confirmation that the response was delivered.
        """
        if self.done:
            raise ValueError("Episode already ended.")

        action = SupportAction(message=f"[respond('{message}')]")
        obs = self._env.step(action)
        self._cumulative_reward += float(obs.reward or 0.0)
        self.done = obs.done

        # Set final reward using grader score (0.0-1.0)
        grader = obs.metadata.get("grader_score", 0.0)
        self.reward = float(grader)

        return "Response delivered to customer. Episode complete."

    # ── Internal ─────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, *args) -> str:
        """Route a tool call through the environment's step method."""
        if self.done:
            raise ValueError("Episode already ended.")

        # Build the bracketed action string the environment expects
        args_str = ", ".join(f"'{a}'" for a in args)
        action_str = f"[{tool_name}({args_str})]"

        action = SupportAction(message=action_str)
        obs = self._env.step(action)

        self._cumulative_reward += float(obs.reward or 0.0)
        self.done = obs.done

        if self.done:
            grader = obs.metadata.get("grader_score", 0.0)
            self.reward = float(grader)

        # Return the feedback message from the environment
        if obs.messages:
            return obs.messages[-1]["content"]
        return f"Tool {tool_name} executed successfully."
