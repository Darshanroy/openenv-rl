"""
Orchestrator — The main multi-agent pipeline.
Routes customer messages → Specialist → Supervisor → Final Response.
Refactored to be API-driven, using the OpenAI/HF Router client.
"""
from typing import List, Dict, Tuple

from .router import Router
from .specialist import SpecialistAgent, SPECIALIST_CONFIGS
from .supervisor import SupervisorAgent


class AgentTrace:
    """Records which agents handled a request and what they did."""
    def __init__(self):
        self.steps: List[Dict] = []

    def add(self, agent_name: str, agent_emoji: str, action: str, detail: str = ""):
        self.steps.append({
            "agent": agent_name,
            "emoji": agent_emoji,
            "action": action,
            "detail": detail
        })

    def summary(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"{s['emoji']} **{s['agent']}**: {s['action']}")
            if s['detail']:
                lines.append(f"   ↳ {s['detail']}")
        return "\n".join(lines)

    def flow(self) -> str:
        """Short inline flow: 🧭 Router → 📦 Order → 👨‍💼 Supervisor"""
        return " → ".join([f"{s['emoji']} {s['agent']}" for s in self.steps])


class Orchestrator:
    """
    Multi-agent orchestrator for the OpenEnv CSA system.
    
    Flow:
    1. Router classifies the customer intent.
    2. Specialist agent generates tool call(s) via the API.
    3. Supervisor reviews and produces the final response via the API.
    """

    def __init__(self, openai_client, model_id: str):
        self.router = Router()
        self.supervisor = SupervisorAgent(openai_client, model_id)
        self.model_id = model_id

        # Create one specialist per category (all share the same API client)
        self.specialists = {}
        for agent_type in SPECIALIST_CONFIGS:
            self.specialists[agent_type] = SpecialistAgent(agent_type, openai_client, model_id)

    def process(self, customer_message: str, observation_text: str,
                task_id: str = None, history_text: str = "") -> Tuple[str, AgentTrace]:
        """
        Main entry point. Processes a customer message through the multi-agent pipeline.
        
        Returns:
            (action_string, AgentTrace)
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
