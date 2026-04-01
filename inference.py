"""
Inference Script — OpenEnv CSA (Multi-Agent Edition)
===================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- This version uses the Multi-Agent Orchestrator (Router -> Specialist -> Supervisor)
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import textwrap
import time
from typing import List, Optional, Dict, Any

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Import our Multi-Agent Brain
from agents.orchestrator import Orchestrator

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
DEBUG = True

# --- Full 15-Task Evaluation Suite ---
TASKS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation",
]


# ==============================================================================
# ENVIRONMENT CLIENT — Talks to the deployed Docker container via HTTP
# ==============================================================================

class SupportEnvClient:
    """Lightweight client that talks to the OpenEnv CSA environment over HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = f"eval_{int(time.time())}"

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

    def feedback(self, message_index: int, feedback_type: str) -> dict:
        """POST /session/feedback/{session_id} — log feedback."""
        url = f"{self.base_url}/session/feedback/{self.session_id}"
        data = {"message_index": message_index, "feedback_type": feedback_type}
        resp = requests.post(url, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """GET /health — check if the environment is live."""
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False


# ==============================================================================
# TASK RUNNER
# ==============================================================================

def run_task(orch: Orchestrator, env: SupportEnvClient, task_id: str) -> float:
    print(f"\n{'='*50}")
    print(f"[EVAL] Task: {task_id}")
    print(f"{'='*50}")

    history_lines: List[str] = []

    try:
        # 1. Reset environment for this task
        obs = env.reset(task_id=task_id)
        # messages_list = obs.get("messages", [])
        # customer_msg = next((m["content"] for m in messages_list if m["role"] == "customer"), "")
        customer_msg = obs.get("prompt", "") # The goal prompt
        
        current_obs = f"Task Start: {customer_msg}"
        done = obs.get("done", False)

        print(f"Goal: {customer_msg}")

        for step in range(1, MAX_STEPS + 1):
            if done:
                print("\n✅ Task concluded by environment.")
                break

            # 2. Multi-Agent Reasoning via Orchestrator
            print(f"\n--- Step {step} ---")
            history_text = "\n".join(history_lines)
            
            # Simulate thinking for better user experience
            print(f"\n[STEP {step}] 🧠 Agent is thinking...")
            time.sleep(1.5)
            
            action_str, trace = orch.process(
                customer_message=customer_msg,
                observation_text=current_obs,
                task_id=task_id,
                history_text=history_text
            )

            # Log Agent Trace to terminal
            print(trace.summary())
            print(f"🚀 **Action Taken**: `{action_str}`")

            # 3. Execute step against the deployed environment
            step_result = env.step(action_str)
            reward = step_result.get("reward", 0.0) or 0.0
            done = step_result.get("done", False)
            step_messages = step_result.get("messages", [])
            last_feedback = step_messages[-1]["content"] if step_messages else "No observation."
            
            current_obs = f"Observation: {last_feedback}"
            history_lines.append(f"Agent: {action_str}")
            history_lines.append(f"System: {last_feedback}")

            print(f"Reward: {reward:+.2f} | Done: {done}")

            if done:
                metadata = step_result.get("metadata", {})
                final_score = metadata.get("grader_score", 0.0)
                exit_reason = metadata.get("exit_reason", "unknown")
                print(f"\n✅ Episode complete. Reason: {exit_reason} | Final Grader Score: {final_score}")
                
                return final_score

    except Exception as e:
        print(f"\n [FATAL ERROR] {e}")
        return 0.0

    print(f"\nReached max steps ({MAX_STEPS}).")
    return 0.0


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY is not set. Set it in .env or as an env var.")
        return

    # 1. Initialize Clients
    client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnvClient(base_url=ENV_URL)
    
    # 2. Initialize Multi-Agent Orchestrator
    orchestrator = Orchestrator(openai_client=client_llm, model_id=MODEL_NAME)

    # 3. Health check
    if not env.health():
        print(f"[ERROR] Environment at {ENV_URL} is not responding. Check your HF Space.")
        return

    print("=" * 60)
    print("OPENENV CSA MULTI-AGENT EVALUATION")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Environment  : {ENV_URL}")
    print("=" * 60)

    results = {}
    for tid in TASKS:
        score = run_task(orchestrator, env, tid)
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
        print(f"  {'AVERAGE EVAL SCORE':25} | {avg:>6.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
