"""
We are importing the orchestrator logic from the agents folder

this script depends on the agents folder for the agentic soo run this script with the agents folder in local  
"""

import os
import sys

# Force UTF-8 stdout so emojis/unicode don't crash on Windows charmap
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
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
# The system pulls these from the environment or a .env file.
# API_BASE_URL: The endpoint for LLM inference (e.g. HF Router or OpenAI)
# API_KEY: Your authentication token
# MODEL_NAME: The specific model ID to use for reasoning
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# ==============================================================================
# ENVIRONMENT CONFIGURATION — Points to the deployed HF Space
# ==============================================================================
# ENV_URL: The public URL where your Dockerized Environment API is running.
ENV_URL = os.getenv("ENV_URL", "https://darshankumarr03-openenv-csa-rl.hf.space")

MAX_STEPS = 8  # Maximum turns allowed per customer interaction
DEBUG = True   # Enable detailed log output

# --- Full 15-Task Evaluation Suite ---
# These task IDs must exist in the environment's mock database (server/db.py)
TASKS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation",
]


# ==============================================================================
# ENVIRONMENT CLIENT — Talks to the deployed Docker container via HTTP
# ==============================================================================

class SupportEnvClient:
    """
    Lightweight client that talks to the OpenEnv CSA environment over HTTP.
    It encapsulates the REST API calls to the remote Hugging Face Space.
    """

    def __init__(self, base_url: str):
        # Ensure base_url doesn't have a trailing slash
        self.base_url = base_url.rstrip("/")
        # Unique session ID for each evaluation run to prevent state collisions
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
    """
    Orchestrates the evaluation of a single customer support task.
    Emits strict [START]/[STEP]/[END] structured output to stdout.

    Returns:
        The final score in [0, 1].
    """
    # Tracking state for structured output
    all_rewards: List[float] = []
    total_steps = 0
    final_score = 0.0
    success = False
    action_str = ""

    # === [START] ===
    print(f"[START] task={task_id} env=CustomerSupport-v1 model={MODEL_NAME}", flush=True)

    try:
        # 1. Reset environment for this specific task
        obs = env.reset(task_id=task_id)
        customer_msg = obs.get("prompt", "")
        current_obs = f"Task Start: {customer_msg}"
        done = obs.get("done", False)
        history_lines: List[str] = []

        # 2. Execute up to MAX_STEPS
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            history_text = "\n".join(history_lines)

            # Agent reasoning
            action_str, trace = orch.process(
                customer_message=customer_msg,
                observation_text=current_obs,
                task_id=task_id,
                history_text=history_text
            )

            # Execute action against the environment
            step_result = env.step(action_str)
            reward = float(step_result.get("reward", 0.0) or 0.0)
            done = step_result.get("done", False)
            step_messages = step_result.get("messages", [])
            last_feedback = step_messages[-1]["content"] if step_messages else "No observation."
            error_msg = step_result.get("last_action_error") or step_result.get("error")

            # Update tracking
            current_obs = f"Observation: {last_feedback}"
            history_lines.append(f"Agent: {action_str}")
            history_lines.append(f"System: {last_feedback}")
            total_steps = step
            all_rewards.append(reward)

            # Sanitize action string: no newlines allowed in structured line
            safe_action = action_str.replace("\n", " ").replace("\r", "")
            done_str = "true" if done else "false"
            error_str = str(error_msg).replace("\n", " ").replace("\r", "") if error_msg else "null"

            # === [STEP] ===
            print(f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

            if done:
                # Use the environment's grader_score (already [0,1] weighted tool coverage)
                metadata = step_result.get("metadata", {})
                grader = float(metadata.get("grader_score", 0.0) or 0.0)
                final_score = max(0.0, min(1.0, grader))
                success = final_score > 0.0
                break

    except Exception as e:
        # On exception: score=0.0, success=false
        error_str = str(e).replace("\n", " ").replace("\r", "")
        print(f"[STEP] step={total_steps + 1} action={action_str} reward=0.00 done=true error={error_str}", flush=True)
        all_rewards.append(0.0)
        total_steps += 1
        final_score = 0.0
        success = False

    # Guarantee final_score is in [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))

    # === [END] — always emitted ===
    print(f"[END] task={task_id} score={final_score:.2f} steps={total_steps}", flush=True)
    print()  # Leave a space after each end

    return final_score


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
