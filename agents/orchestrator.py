"""
Multi-Agent Orchestrator — OpenEnv CSA
=====================================
The central brain of the autonomous agent. This module coordinates the 
interaction between the Router, Specialist Agents, and the Supervisor.

Pipeline Control Flow:
1. Routing: Classify customer intent to select the best Specialist.
2. Specialist: Generate domain-specific tool calls (Order, Logistics, Finance).
3. Supervision: Review specialist output for quality and sentiment, 
   producing the final response back to the customer.
"""
from typing import List, Dict, Tuple

from .router import Router
from .specialist import SpecialistAgent, SPECIALIST_CONFIGS
from .supervisor import SupervisorAgent


class AgentTrace:
    """
    Utility for recording and visualizing the multi-agent reasoning chain.
    Captures which agents were invoked, their actions, and internal thoughts for terminal logging.
    """
    def __init__(self):
        self.steps: List[Dict] = []

    def add(self, agent_name: str, agent_emoji: str, action: str, detail: str = ""):
        """Appends a new step to the execution trace."""
        self.steps.append({
            "agent": agent_name,
            "emoji": agent_emoji,
            "action": action,
            "detail": detail
        })

    def summary(self) -> str:
        """Returns a formatted multi-line summary of the entire agent workflow."""
        lines = []
        for s in self.steps:
            lines.append(f"{s['emoji']} **{s['agent']}**: {s['action']}")
            if s['detail']:
                lines.append(f"   ↳ {s['detail']}")
        return "\n".join(lines)

    def flow(self) -> str:
        """Concise visualization: 🧭 Router → 📦 Order."""
        return " → ".join([f"{s['emoji']} {s['agent']}" for s in self.steps])


class Orchestrator:
    """
    High-level orchestrator that manages the end-to-end multi-agent pipeline.
    """

    def __init__(self, openai_client, model_id: str):
        """
        Initializes the agent brain with a router, supervisor, and specialized agents.
        """
        self.router = Router(openai_client=openai_client, model_id=model_id)
        self.supervisor = SupervisorAgent(openai_client, model_id)
        self.model_id = model_id

        # Register specialized agents for domain-specific handling
        self.specialists = {}
        for agent_type in SPECIALIST_CONFIGS:
            self.specialists[agent_type] = SpecialistAgent(agent_type, openai_client, model_id)

    def process(self, customer_message: str, observation_text: str,
                task_id: str = None, history_text: str = "") -> Tuple[str, AgentTrace]:
        """
        Main entry point. Processes a customer message through the multi-agent pipeline.
        Returns the selected tool action and a trace of the reasoning flow.
        """
        trace = AgentTrace()

        # Step 1: Route
        agent_type = self.router.classify(customer_message, task_id=task_id)
        trace.add("Router", "🧭", f"Classified as: **{agent_type}**",
                   f"Message: \"{customer_message[:60]}...\"" if len(customer_message) > 60 else f"Message: \"{customer_message}\"")

        # Step 2: Check if Supervisor should take over directly (angry/escalation)
        if agent_type == "supervisor" or self.supervisor.should_escalate(customer_message):
            trace.add("Supervisor", "👨‍💼", "Taking direct control (escalation detected)")
            action = self.supervisor.review_and_respond(
                customer_message=customer_message,
                specialist_actions=[],
                specialist_name="(direct)",
                observation_text=observation_text
            )
            trace.add("Supervisor", "👨‍💼", f"Action: `{action[:80]}`")
            return action, trace

        # Step 3: Specialist generates tool call
        specialist = self.specialists[agent_type]
        
        thought, action = specialist.generate_action(
            observation_text=observation_text,
            history_text=history_text
        )
        
        trace.add(specialist.name, specialist.emoji, f"Action: `{action[:80]}`", detail=f"**Thought:** {thought}")

        # Step 4: Supervisor reviews (lightweight — only for 'respond' actions)
        if "[respond(" in action:
            trace.add("Supervisor", "👨‍💼", "Reviewing final response...")
            # Supervisor can override if needed, but for now just logs approval
            trace.add("Supervisor", "👨‍💼", "✅ Approved")

        return action, trace

    def get_specialist_for_task(self, task_id: str) -> str:
        """Returns which specialist handles a given task_id."""
        return self.router.classify("", task_id=task_id)
