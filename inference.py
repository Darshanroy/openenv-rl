import os
import re
import time
import requests
import textwrap
from typing import List, Optional, Dict
from openai import OpenAI

# ==============================================================================
# MANDATORY CONFIGURATION
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# LOCAL CONFIGURATION
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS = 6
TEMPERATURE = 0.1
MAX_TOKENS = 512
FALLBACK_ACTION = "[respond('I encountered an error. How can I help you?')]"

# --- Task Tiers ---
TASKS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
]

# Regex for extracting tool calls in the format [tool_name('param')]
ACTION_PATTERN = re.compile(r"\[[A-Za-z_]+\(.*?\)\]", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Customer Support AI. Your goal is to resolve the customer's issue efficiently.
    You have access to tools that you call by writing [tool_name('param')].
    
    Available tools:
    - get_order(order_id)
    - track_shipment(order_id)
    - validate_coupon(code)
    - reset_password(email)
    - initiate_refund(order_id)
    - ask_proof(order_id)
    - investigate_missing(order_id)
    - escalate_to_human(reason)
    - respond(message)

    Internal Reasoning: Always start your response with <thought>...</thought>
    Final Action: End your response with exactly ONE tool call.
    Example: [get_order('ORD-101')]
    """
).strip()

def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])

def build_user_prompt(step: int, observation_text: str, history: List[str]) -> str:
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Previous History:
        {build_history_lines(history)}

        Current Observation:
        {observation_text}

        Reply with exactly one tool call in brackets [tool_name('param')].
        """
    ).strip()
    return prompt

def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Search for the last tool call in the text
    matches = ACTION_PATTERN.findall(response_text)
    if matches:
        return matches[-1].strip()

    return FALLBACK_ACTION

def run_task(client: OpenAI, task_id: str) -> float:
    print(f"\n[EVAL] Evaluating Task: {task_id}")
    session_id = f"eval_{task_id}_{int(time.time())}"
    
    # 1. Reset Environment
    try:
        res = requests.post(
            f"{ENV_URL}/session/reset", 
            json={"session_id": session_id, "task_id": task_id},
            timeout=10
        ).json()
    except Exception as e:
        print(f"   [ENV ERROR] Reset failed: {e}")
        return 0.0

    obs_text = res["messages"][-1]["content"] if res.get("messages") else "Hello, how can I help?"
    history: List[str] = []
    done = False
    turn = 0
    final_score = 0.0

    # 2. Step Loop
    while not done and turn < MAX_STEPS:
        turn += 1
        user_prompt = build_user_prompt(turn, obs_text, history)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"   [LLM ERROR] Step {turn}: {e}")
            response_text = FALLBACK_ACTION

        action_str = parse_model_action(response_text)
        print(f"   Turn {turn} Agent: {action_str}")

        # Step into environment
        try:
            step_res = requests.post(
                f"{ENV_URL}/session/step/{session_id}",
                json={"message": action_str},
                timeout=10
            ).json()
        except Exception as e:
            print(f"   [ENV ERROR] Step failed: {e}")
            break

        # Record history
        history.append(f"Turn {turn}: {action_str}")
        
        if step_res.get("messages"):
            obs_text = step_res["messages"][-1]["content"]
        
        if step_res.get("done"):
            final_score = step_res.get("metadata", {}).get("grader_score", 0.0)
            print(f"   [SUCCESS] Task Complete. Score: {final_score}")
            done = True

    return final_score

def main():
    if not API_KEY:
        print("[ERROR] API_KEY (HF_TOKEN) is not set.")
        # We don't exit here so the user can see the structure, 
        # but in a real run it would fail.

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY if API_KEY else "not-set")

    print("="*60)
    print("OPENENV EVALUATION (Submission Ready)")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print("="*60)

    results = {}
    for tid in TASKS:
        try:
            score = run_task(client, tid)
            results[tid] = score
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"   [WARNING] Fatal Error in {tid}: {e}")
            results[tid] = 0.0

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
