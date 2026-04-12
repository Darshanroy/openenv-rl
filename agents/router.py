"""
LLM-Powered Router Agent — Classifies customer intent using the project's LLM.

Upgraded from pure keyword matching to real Natural Language Understanding.
Uses the same Qwen model configured in .env (via OpenAI-compatible API) to
classify customer messages into the correct specialist category.

Falls back to keyword scoring if the LLM is unavailable or returns garbage.
"""
import re
import json
import logging

logger = logging.getLogger(__name__)

# Valid agent categories the router can dispatch to
VALID_CATEGORIES = ["order", "logistics", "finance", "supervisor"]

# LLM system prompt for intent classification — concise to minimize latency
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for an e-commerce customer support system.

Given a customer message, classify it into EXACTLY ONE of these categories:

- "order": Order status, cancellations, payment issues, coupon codes, password resets, account problems.
- "logistics": Shipping, delivery tracking, address changes, rescheduling, delays, missing packages.
- "finance": Refunds, returns, damaged/broken items, double charges, money back requests.
- "supervisor": Angry/abusive customers, requests for a manager, escalation demands, threats.

RULES:
1. If the customer is angry AND asking about a specific issue (refund, delivery), classify by the EMOTION first — use "supervisor" if the tone is hostile, threatening, or demands a manager.
2. If the message mentions damaged/broken items or refund requests, use "finance".
3. If the message is about where a package is or delivery timing, use "logistics".
4. Default to "order" if ambiguous.

Respond with ONLY a JSON object: {"category": "<one of: order, logistics, finance, supervisor>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

# Keyword fallback table — used when LLM is unavailable
KEYWORD_FALLBACK = {
    "order": [
        "order", "status", "cancel", "payment", "coupon", "password",
        "account", "login", "forgot", "apply coupon", "reset password"
    ],
    "logistics": [
        "shipping", "delivery", "track", "address", "reschedule",
        "delay", "late", "missing", "delivered but", "not here",
        "change address", "slot", "where is", "hold up"
    ],
    "finance": [
        "refund", "return", "damaged", "broken", "money back",
        "charged twice", "double charge", "shattered", "crushed"
    ],
    "supervisor": [
        "manager", "supervisor", "escalate", "sue",
        "pathetic", "terrible", "worst", "furious", "complain"
    ]
}


class Router:
    """
    LLM-powered intent classifier with keyword fallback.

    Primary path: Sends the customer message to the LLM for semantic understanding.
    Fallback path: Uses keyword scoring if the LLM call fails or times out.
    """

    def __init__(self, openai_client=None, model_id: str = None):
        """
        Args:
            openai_client: An OpenAI-compatible client instance (same one used by Specialists).
            model_id: The model identifier (e.g., 'Qwen/Qwen2.5-7B-Instruct').
        """
        self.client = openai_client
        self.model_id = model_id
        self._llm_available = openai_client is not None and model_id is not None

    def classify(self, message: str, task_id: str = None) -> str:
        """
        Determines the most appropriate agent category for a given message.

        Logic:
        1. LLM Classification: Send the message to the LLM for semantic NLU.
        2. Validation: Ensure the LLM returned a valid category.
        3. Fallback: Use keyword scoring if the LLM fails.
        """
        # Primary path: LLM-powered classification
        if self._llm_available and message.strip():
            try:
                category = self._llm_classify(message, task_id)
                if category in VALID_CATEGORIES:
                    logger.info(f"LLM Router → {category} (message: {message[:50]}...)")
                    return category
                else:
                    logger.warning(f"LLM returned invalid category '{category}', falling back to keywords")
            except Exception as e:
                logger.warning(f"LLM Router failed ({e}), falling back to keywords")

        # Fallback path: keyword scoring
        return self._keyword_classify(message)

    def _llm_classify(self, message: str, task_id: str = None) -> str:
        """
        Uses the LLM to semantically classify the customer's intent.
        Returns one of: 'order', 'logistics', 'finance', 'supervisor'.
        """
        # Build the user prompt with optional task context
        user_prompt = f"Customer message: \"{message}\""
        if task_id:
            user_prompt += f"\n(Internal task reference: {task_id})"

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,   # Deterministic classification
            max_tokens=100,    # Short response — just the JSON
        )

        response = (completion.choices[0].message.content or "").strip()

        # Parse the JSON response
        category = self._parse_llm_response(response)
        return category

    def _parse_llm_response(self, response: str) -> str:
        """
        Extracts the category from the LLM's JSON response.
        Handles various formats the LLM might use.
        """
        # Try direct JSON parse first
        try:
            # Strip markdown code fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```(?:json)?\s*', '', clean)
                clean = re.sub(r'\s*```$', '', clean)

            data = json.loads(clean)
            category = data.get("category", "").lower().strip()
            confidence = data.get("confidence", 0.0)
            reasoning = data.get("reasoning", "")
            logger.info(f"LLM NLU: category={category}, confidence={confidence}, reason={reasoning[:60]}")
            return category
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for category name in raw text
        response_lower = response.lower()
        for cat in VALID_CATEGORIES:
            if cat in response_lower:
                return cat

        return "unknown"

    def _keyword_classify(self, message: str) -> str:
        """
        Legacy keyword-based classification. Used as fallback when LLM is unavailable.
        Scores each category by keyword match count and returns the best match.
        """
        msg_lower = message.lower()
        scores = {}
        for agent_type, keywords in KEYWORD_FALLBACK.items():
            score = sum(1 for kw in keywords if kw.lower() in msg_lower)
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
