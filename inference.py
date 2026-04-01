import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Import our new Docker-integrated Environment Wrapper
from support_env import SupportAction, SupportEnv

# Load local environment variables from .env
load_dotenv()

# ==============================================================================
# MANDATORY CONFIGURATION
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# LOCAL CONFIGURATION
IMAGE_NAME = "openenv-csa-env:latest"  # The Docker image built from Dockerfile.env
MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 512
FALLBACK_ACTION = "[respond('I encountered an error. How can I help you?')]"

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action|action string)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"\[[A-Za-z_]+\(.*?\)\]", re.DOTALL)

# --- Tasks to Evaluate ---
TASKS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Customer Support AI. Your goal is to resolve the customer's issue efficiently.
    You control a support system. Reply with exactly ONE action string in brackets [tool_name('param')].

    ### PROTOCOLS (MANDATORY):
    1. **Damaged/Broken Items**: You MUST call `ask_proof(order_id)` before you can refund.
    2. **Address Changes**: You MUST call `get_order(order_id)` first to check if it has already shipped.
    3. **Missing/Delayed Orders**: Always call `track_shipment(order_id)` before calling `investigate_missing`.
    4. **Cancellations**: Check order status first via `get_order`. If already delivered, you cannot cancel, you must start a return.

    Available tools:
    - get_order(order_id): Get details and status of an order.
    - track_shipment(order_id): Get real-time tracking data.
    - validate_coupon(code): Check if a discount code is valid.
    - reset_password(email): Send a password reset link.
    - initiate_refund(order_id): Refund the customer (only after validation).
    - ask_proof(order_id): Request a photo of damaged items (Required for damaged claims).
    - investigate_missing(order_id): Open a case for a lost package.
    - escalate_to_human(reason): Forward to a human manager.
    - respond(message): Send your final answer to the customer.

    Internal Reasoning: Always start your response with <thought>...</thought>
    Final Action: End your response with exactly ONE tool call in brackets.
    Example: [get_order('ORD-101')]
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-6:])


def extract_screenshot_uri(observation) -> Optional[str]:
    """Helper for future vision-language models (Optional)."""
    if observation.screenshot is None:
        return None
    # If vision is added later, convert observation.screenshot to base64 here.
    return None


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = getattr(observation, "goal", "(not provided)")
    error_note = "Yes" if observation.last_action_error else "No"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Previous History:
        {build_history_lines(history)}
        Last action error: {error_note}

        Reply with exactly one tool call in brackets [tool_name('param')].
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Search for the last tool call in the text [tool('param')]
    matches = ACTION_PATTERN.findall(response_text)
    if matches:
        return matches[-1].strip()

    # Recovery: Check for naked tool calls without brackets
    for tool in ["get_order", "track_shipment", "validate_coupon", "reset_password", "initiate_refund", "ask_proof", "investigate_missing", "escalate_to_human", "respond"]:
        if f"{tool}(" in response_text:
            start_idx = response_text.find(f"{tool}(")
            end_idx = response_text.find(")", start_idx)
            if end_idx != -1:
                return f"[{response_text[start_idx:end_idx+1].strip()}]"

    if DEBUG:
        print(f"   [DEBUG RAW RESPONSE] Parser failed to find brackets in:\n{textwrap.indent(response_text, '      > ')}")
    
    return FALLBACK_ACTION


def run_task(client: OpenAI, env: SupportEnv, task_id: str) -> float:
    print(f"\n[EVAL] Evaluating Task: {task_id}")
    history: List[str] = []
    
    try:
        # Reset the Docker-backed environment
        result = env.reset(task_id=task_id)
        observation = result.observation
        print(f"Episode goal: {observation.goal}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, observation, history)
            user_content = [{"type": "text", "text": user_prompt}]
            
            # Screenshot future-proofing (Official Flow compatibility)
            screenshot_uri = extract_screenshot_uri(observation)
            if screenshot_uri:
                user_content.append({"type": "image_url", "image_url": {"url": screenshot_uri}})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            # LLM Call with Retries for Reliability
            response_text = ""
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    response_text = completion.choices[0].message.content or ""
                    break
                except Exception as exc:
                    if "429" in str(exc) or "rate_limit" in str(exc).lower():
                        time_sleep = 10 * (attempt + 1)
                        print(f"   [RATE LIMIT] Waiting {time_sleep}s...")
                        import time; time.sleep(time_sleep)
                    else:
                        print(f"   [LLM ERROR] Step {step}: {exc}")
                        response_text = FALLBACK_ACTION
                        break

            action_str = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_str}")

            # Execute step in Docker container
            result = env.step(SupportAction(action_str=action_str))
            observation = result.observation

            reward = result.reward or 0.0
            error_flag = " ERROR" if observation.last_action_error else ""
            history_line = f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            history.append(history_line)
            
            print(f"  Reward: {reward:+.2f} | Done: {result.done} | Error: {observation.last_action_error}")

            if result.done:
                final_score = result.metadata.get("grader_score", 0.0)
                print(f"   [SUCCESS] Episode complete. Final Grader Score: {final_score}")
                return final_score

    except Exception as e:
        print(f"   [FATAL ERROR] {e}")
        return 0.0

    return 0.0


def main() -> None:
    if not API_KEY:
        print("[ERROR] API_KEY (HF_TOKEN) is not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY if API_KEY else "not-set")

    # Start the Docker-based environment (Standardized Lifecycle)
    try:
        env = SupportEnv.from_docker_image(image=IMAGE_NAME)
    except Exception as e:
        print(f"[RETRY] Attempting to find local image {IMAGE_NAME}...")
        raise e

    print("="*60)
    print("OPENENV EVALUATION (Official Docker Flow)")
    print(f"Model ID: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    print("="*60)

    results = {}
    try:
        for tid in TASKS:
            score = run_task(client, env, tid)
            results[tid] = score
    finally:
        # Mandatory Cleanup: Kills the environment container automatically
        env.close()

    # Final Summary
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    for tid, score in results.items():
        print(f"{tid:25} | {score:>6.2f}")
    
    if results:
        avg = sum(results.values()) / len(results)
        print("-" * 60)
        print(f"{'AVERAGE SCORE':25} | {avg:>6.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
