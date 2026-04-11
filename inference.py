"""
OpenEnv CSA — Multi-Agent Inference & Evaluation Script.

This is the main entry point for evaluating the Customer Support Agent across
all 15 e-commerce tasks. It connects the local Multi-Agent Orchestrator (Router
→ Specialist → Supervisor) to the remote OpenEnv Environment API.

Dependencies:
    - agents/orchestrator.py  (local multi-agent brain)
    - A running OpenEnv Environment (Docker container on HF Space)

Scoring Protocol:
    - All task scores are strictly between 0 and 1 (never 0.0 or 1.0).
    - The final score for each task comes from the environment's grader_score,
      which is returned in the step metadata when done=True.
    - The grader_score is a weighted sum of which correct tools the agent used,
      normalized to the (0.01, 0.99) range via the environment's internal logic.
"""

import os
import sys

# Force UTF-8 stdout so emojis/unicode don't crash on Windows charmap
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import time
from typing import List, Optional, Dict, Any

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Import the Multi-Agent Orchestrator (Router → Specialist → Supervisor)
from agents.orchestrator import Orchestrator

# Load local environment variables (.env file)
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# LLM Configuration — pulled from .env or system environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# Environment Configuration — points to the deployed HF Space
ENV_URL = os.getenv("ENV_URL", "https://darshankumarr03-openenv-csa-rl.hf.space")

# Evaluation parameters
MAX_STEPS = 8   # Maximum agent turns per task (matches environment's max_turns)

# Full 15-Task Evaluation Suite across 3 difficulty tiers
TASKS = [
    # Easy (5 tasks) — single-tool resolution
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account", "easy_cancel",
    # Medium (5 tasks) — multi-tool workflows
    "medium_delay", "medium_address", "medium_reschedule", "medium_return", "medium_double_charge",
    # Hard (5 tasks) — protocol enforcement, escalation, emotional handling
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation",
]


# ==============================================================================
# ENVIRONMENT CLIENT — Communicates with the remote Docker container via HTTP
# ==============================================================================

class SupportEnvClient:
    """
    Lightweight REST client for the OpenEnv CSA environment.
    Talks to the deployed Hugging Face Space over HTTP.

    Endpoints:
        POST /session/reset         — Start a new episode
        POST /session/step/{id}     — Execute one agent action
        POST /session/feedback/{id} — Log RLHF feedback
        GET  /health                — Liveness probe
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Unique session ID per evaluation run to prevent state collisions
        self.session_id = f"eval_{int(time.time())}"

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new episode for the given task. Returns initial observation."""
        resp = requests.post(
            f"{self.base_url}/session/reset",
            json={"session_id": self.session_id, "task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action_str: str) -> Dict[str, Any]:
        """Execute one agent action. Returns observation, reward, done, metadata."""
        resp = requests.post(
            f"{self.base_url}/session/step/{self.session_id}",
            json={"message": action_str},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def feedback(self, message_index: int, feedback_type: str) -> dict:
        """Log RLHF feedback (thumbs_up/thumbs_down) for a message."""
        url = f"{self.base_url}/session/feedback/{self.session_id}"
        data = {"message_index": message_index, "feedback_type": feedback_type}
        resp = requests.post(url, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Check if the environment API is live and responding."""
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False


# ==============================================================================
# TASK RUNNER — Evaluates a single task end-to-end
# ==============================================================================

def run_task(orch: Orchestrator, env: SupportEnvClient, task_id: str) -> float:
    """
    Run a single customer support task through the multi-agent pipeline.

    The agent interacts with the environment for up to MAX_STEPS turns.
    The final score is the environment's grader_score (strictly in (0, 1)),
    which measures how many of the required tools the agent correctly used.

    Emits structured [START]/[STEP]/[END] lines to stdout for parsing.

    Args:
        orch: The multi-agent orchestrator instance.
        env: The environment REST client.
        task_id: The scenario to evaluate (e.g., "easy_status").

    Returns:
        The final grader score, strictly between 0.01 and 0.99.
    """
    # -- State tracking --
    all_rewards: List[float] = []   # Step-by-step rewards (each in (0, 1))
    total_steps = 0
    final_score = 0.0
    action_str = ""

    # === [START] — marks the beginning of a task evaluation ===
    print(f"[START] task={task_id} env=CustomerSupport-v1 model={MODEL_NAME}", flush=True)

    try:
        # 1. Reset the environment for this specific task
        obs = env.reset(task_id=task_id)
        customer_msg = obs.get("prompt", "")
        current_obs = f"Task Start: {customer_msg}"
        done = obs.get("done", False)
        history_lines: List[str] = []

        # 2. Agent-Environment interaction loop (up to MAX_STEPS turns)
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build conversation history for context
            history_text = "\n".join(history_lines)

            # Multi-agent reasoning: Router → Specialist → Supervisor
            action_str, trace = orch.process(
                customer_message=customer_msg,
                observation_text=current_obs,
                task_id=task_id,
                history_text=history_text
            )

            # Execute the agent's action against the environment
            step_result = env.step(action_str)

            # Parse the environment's response
            # reward: sigmoid-normalized step reward, strictly in (0, 1)
            reward = float(step_result.get("reward", 0.0) or 0.0)
            done = step_result.get("done", False)
            step_messages = step_result.get("messages", [])
            last_feedback = step_messages[-1]["content"] if step_messages else "No observation."
            error_msg = step_result.get("last_action_error") or step_result.get("error")

            # Update tracking for next iteration
            current_obs = f"Observation: {last_feedback}"
            history_lines.append(f"Agent: {action_str}")
            history_lines.append(f"System: {last_feedback}")
            total_steps = step
            all_rewards.append(reward)

            # Sanitize action string for structured log output (no newlines)
            safe_action = action_str.replace("\n", " ").replace("\r", "")
            done_str = "true" if done else "false"
            error_str = str(error_msg).replace("\n", " ").replace("\r", "") if error_msg else "null"

            # === [STEP] — one line per agent-environment interaction ===
            print(f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

            if done:
                # Extract the grader_score from the environment's metadata.
                # This is the authoritative task completion score, already
                # normalized to (0.01, 0.99) by the environment's internal logic.
                metadata = step_result.get("metadata", {})
                final_score = float(metadata.get("grader_score", 0.01))
                break

    except Exception as e:
        # On exception: emit an error step and set score to minimum
        error_str = str(e).replace("\n", " ").replace("\r", "")
        print(f"[STEP] step={total_steps + 1} action={action_str} reward=0.00 done=true error={error_str}", flush=True)
        total_steps += 1
        final_score = 0.01  # Minimum score on failure (never 0.0)

    # If the loop ended without the environment signaling done (e.g., max_turns
    # reached on the client side), no grader_score was obtained — use minimum.
    if final_score == 0.0:
        final_score = 0.01

    # Final safety clamp: guarantee score is strictly in (0.0, 1.0)
    final_score = max(0.01, min(0.99, final_score))

    # === [END] — always emitted, marks the end of a task evaluation ===
    print(f"[END] task={task_id} score={final_score:.2f} steps={total_steps}", flush=True)
    print()

    return final_score


# ==============================================================================
# MAIN — Run the full 15-task evaluation suite
# ==============================================================================

def main() -> None:
    """
    Entry point for the OpenEnv CSA evaluation.
    Runs all 15 tasks and computes per-tier and overall averages.
    All scores are strictly between 0 and 1 (never 0.0 or 1.0).
    """
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY is not set. Set it in .env or as an env var.")
        return

    # 1. Initialize clients
    client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnvClient(base_url=ENV_URL)

    # 2. Initialize the Multi-Agent Orchestrator
    orchestrator = Orchestrator(openai_client=client_llm, model_id=MODEL_NAME)

    # 3. Health check — ensure the environment is reachable
    if not env.health():
        print(f"[ERROR] Environment at {ENV_URL} is not responding. Check your HF Space.")
        return

    print("=" * 60)
    print("OPENENV CSA MULTI-AGENT EVALUATION")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Environment  : {ENV_URL}")
    print(f"  Tasks        : {len(TASKS)}")
    print(f"  Score Range  : (0.01, 0.99) — strictly between 0 and 1")
    print("=" * 60)

    # 4. Run all 15 tasks and collect scores
    results: Dict[str, float] = {}
    for tid in TASKS:
        score = run_task(orchestrator, env, tid)
        results[tid] = score

    # ══════════════════════════════════════════════════════════════
    #  FINAL EVALUATION SUMMARY — Per-tier and overall averages
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    # Group tasks by difficulty tier for granular reporting
    tiers = {
        "Easy":   [t for t in results if t.startswith("easy_")],
        "Medium": [t for t in results if t.startswith("medium_")],
        "Hard":   [t for t in results if t.startswith("hard_")],
    }

    # Print per-task scores
    for tid, score in results.items():
        print(f"  {tid:25} | {score:>6.2f}")

    print("-" * 60)

    # Compute and display per-tier averages
    for tier_name, tier_tasks in tiers.items():
        if tier_tasks:
            tier_scores = [results[t] for t in tier_tasks]
            tier_avg = sum(tier_scores) / len(tier_scores)
            print(f"  {tier_name + ' Avg':25} | {tier_avg:>6.4f}  ({len(tier_tasks)} tasks)")

    print("-" * 60)

    # Compute and display overall average
    if results:
        all_scores = list(results.values())
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"  {'OVERALL AVERAGE':25} | {overall_avg:>6.4f}  ({len(all_scores)} tasks)")
    print("=" * 60)


if __name__ == "__main__":
    main()
