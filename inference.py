"""
Inference Script — OpenEnv CSA
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

"""

import os
import re
import textwrap
import time
from typing import List, Optional, Dict, Any

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# ==============================================================================
# MANDATORY CONFIGURATION
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# ==============================================================================
# ENVIRONMENT CONFIGURATION — Points to the deployed HF Space
# ==============================================================================
ENV_URL = os.getenv("ENV_URL", "https://darshankumarr03-openenv-csa-rl.hf.space")

MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 512
FALLBACK_ACTION = "[respond('I encountered an error. How can I help you?')]"

DEBUG = True
ACTION_PATTERN = re.compile(r"\[[A-Za-z_]+\(.*?\)\]", re.DOTALL)

# --- All 15 Tasks ---
TASKS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation",
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Customer Support AI. Your goal is to resolve the customer's issue efficiently.
    You control a support system. Reply with exactly ONE action string in brackets [tool_name('param')].

    ### PROTOCOLS (MANDATORY):
    1. **Damaged/Broken Items**: You MUST call `ask_proof(order_id)` before you can refund.
    2. **Address Changes**: You MUST call `get_order(order_id)` first to check if it has already shipped.
    3. **Missing/Delayed Orders**: Always call `track_shipment(order_id)` before calling `investigate_missing`.
    4. **Cancellations**: Check order status first via `get_order`. If already delivered, you cannot cancel.

    Available tools:
    - get_order(order_id): Get details and status of an order.
    - track_shipment(order_id): Get real-time tracking data.
    - validate_coupon(code): Check if a discount code is valid.
    - reset_password(email): Send a password reset link.
    - initiate_refund(order_id): Refund the customer (only after validation).
    - ask_proof(order_id): Request a photo of damaged items.
    - investigate_missing(order_id): Open a case for a lost package.
    - escalate_to_human(reason): Forward to a human manager.
    - respond(message): Send your final answer to the customer.
    - cancel_order(order_id): Cancel a pending order.
    - update_address(order_id, new_address): Change delivery address.
    - check_delivery_slot(order_id): Check available delivery slots.
    - reschedule_delivery(order_id, slot): Reschedule delivery.
    - validate_return(order_id): Check if an item is eligible for return.
    - create_return_request(order_id): Create a return request.

    Internal Reasoning: Always start your response with <thought>...</thought>
    Final Action: End your response with exactly ONE tool call in brackets.
    Example: [get_order('ORD-101')]
    """
).strip()


# ==============================================================================
# ENVIRONMENT CLIENT — Talks to the deployed Docker container via HTTP
# ==============================================================================

class SupportEnvClient:
    """Lightweight client that talks to the OpenEnv CSA environment over HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = f"inference_{int(time.time())}"

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /session/reset — start a new episode."""
        resp = requests.post(
            f"{self.base_url}/session/reset",
            json={"session_id": self.session_id, "task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action_str: str) -> Dict[str, Any]:
        """POST /session/step/{session_id} — execute one action."""
        resp = requests.post(
            f"{self.base_url}/session/step/{self.session_id}",
            json={"message": action_str},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """GET /health — check if the environment is live."""
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False


# ==============================================================================
# PROMPT BUILDING & ACTION PARSING
# ==============================================================================

def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-6:])


def build_user_prompt(step: int, goal: str, history: List[str], last_error: bool) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Previous History:
        {build_history_lines(history)}
        Last action error: {"Yes" if last_error else "No"}

        Reply with exactly one tool call in brackets [tool_name('param')].
        """
    ).strip()


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Find bracketed tool calls [tool('param')]
    matches = ACTION_PATTERN.findall(response_text)
    if matches:
        return matches[-1].strip()

    # Recovery: naked tool calls without brackets
    known_tools = [
        "get_order", "track_shipment", "validate_coupon", "reset_password",
        "initiate_refund", "ask_proof", "investigate_missing", "escalate_to_human",
        "respond", "cancel_order", "update_address", "check_delivery_slot",
        "reschedule_delivery", "validate_return", "create_return_request",
    ]
    for tool in known_tools:
        if f"{tool}(" in response_text:
            start = response_text.find(f"{tool}(")
            end = response_text.find(")", start)
            if end != -1:
                return f"[{response_text[start:end+1].strip()}]"

    if DEBUG:
        print(f"   [DEBUG] Parser failed:\n{textwrap.indent(response_text, '      > ')}")

    return FALLBACK_ACTION


# ==============================================================================
# TASK RUNNER
# ==============================================================================

def run_task(client_llm: OpenAI, env: SupportEnvClient, task_id: str) -> float:
    print(f"\n{'='*50}")
    print(f"[EVAL] Task: {task_id}")
    print(f"{'='*50}")

    history: List[str] = []

    try:
        # Reset environment for this task
        obs = env.reset(task_id=task_id)
        goal = obs.get("prompt", "(not provided)")
        messages_list = obs.get("messages", [])
        customer_msg = next((m["content"] for m in messages_list if m["role"] == "customer"), goal)
        done = obs.get("done", False)
        last_error = False

        print(f"Episode goal: {customer_msg}")

        for step in range(1, MAX_STEPS + 1):
            if done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, customer_msg, history, last_error)
            user_content = [{"type": "text", "text": user_prompt}]

            llm_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            # LLM Call with retry for rate limits
            response_text = ""
            for attempt in range(3):
                try:
                    completion = client_llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=llm_messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    break
                except Exception as exc:
                    if "429" in str(exc) or "rate_limit" in str(exc).lower():
                        wait = 10 * (attempt + 1)
                        print(f"   [RATE LIMIT] Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"   [LLM ERROR] Step {step}: {exc}")
                        response_text = FALLBACK_ACTION
                        break

            action_str = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_str}")

            # Execute step against the deployed environment
            step_result = env.step(action_str)
            reward = step_result.get("reward", 0.0) or 0.0
            done = step_result.get("done", False)
            step_messages = step_result.get("messages", [])
            last_feedback = step_messages[-1]["content"] if step_messages else ""
            last_error = "Error" in last_feedback or "Unknown" in last_feedback

            error_flag = " ERROR" if last_error else ""
            history_line = f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            history.append(history_line)

            print(
                f"  Reward: {reward:+.2f} | Done: {done} | "
                f"Last action error: {last_error}"
            )

            if done:
                final_score = step_result.get("metadata", {}).get("grader_score", 0.0)
                print(f"Episode complete. Grader Score: {final_score}")
                return final_score

    except Exception as e:
        print(f"   [FATAL ERROR] {e}")
        return 0.0

    print(f"Reached max steps ({MAX_STEPS}).")
    return 0.0


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY is not set. Set it in .env or as an env var.")

    # LLM Client (Official: OpenAI Client)
    client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "not-set")

    # Environment Client (connects to deployed Docker container on HF Space)
    env = SupportEnvClient(base_url=ENV_URL)

    # Health check
    if not env.health():
        print(f"[WARNING] Environment at {ENV_URL} is not responding. Proceeding anyway...")

    print("=" * 60)
    print("OPENENV CSA EVALUATION")
    print(f"  LLM Endpoint : {API_BASE_URL}")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Environment  : {ENV_URL}")
    print("=" * 60)

    results = {}
    for tid in TASKS:
        score = run_task(client_llm, env, tid)
        results[tid] = score

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    for tid, score in results.items():
        print(f"  {tid:25} | {score:>6.2f}")

    if results:
        avg = sum(results.values()) / len(results)
        print("-" * 60)
        print(f"  {'AVERAGE SCORE':25} | {avg:>6.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
