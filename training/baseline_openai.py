"""
OpenAI API Baseline Inference Script
=====================================
Uses the OpenAI API client to run a model against the OpenEnv CSA environment.
Reads API credentials from environment variables (OPENAI_API_KEY).
Produces reproducible baseline scores on all 15 tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_SERVER_URL=http://127.0.0.1:8000  # optional, defaults to localhost
    python training/baseline_openai.py
"""
import os
import sys
import csv
import re
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from my_env.client import SupportEnvClient
from my_env.models import SupportAction

# Configuration
ENV_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TURNS = 6

SCENARIOS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account",
    "easy_cancel",
    "medium_delay", "medium_address", "medium_reschedule", "medium_return",
    "medium_double_charge",
    "hard_refund", "hard_damaged", "hard_missing", "hard_angry",
    "hard_escalation"
]

SYSTEM_PROMPT = """You are an E-Commerce Customer Support Agent.
You interact with the database using tool calls formatted in brackets: [tool_name('param')]

Available tools:
  get_order(order_id) - Fetch order details
  cancel_order(order_id) - Cancel pending orders
  track_shipment(order_id) - Shipment tracking
  update_address(order_id, new_address) - Change delivery address
  check_delivery_slot(order_id) - Available time slots
  reschedule_delivery(order_id, slot) - Reschedule delivery
  investigate_missing(order_id) - Open missing item case
  validate_return(order_id) - Check return eligibility
  ask_proof(order_id) - Request damage photo evidence
  create_return_request(order_id) - Create return label
  initiate_refund(order_id) - Reverse payment
  validate_coupon(code) - Check coupon validity
  reset_password(email) - Send password reset
  escalate_to_human(reason) - Escalate to manager
  respond(message) - Send final response to customer

Rules:
- Always start by looking up the order with get_order().
- For damaged items, ALWAYS ask_proof() before refunding.
- When done, use [respond('your message')] to close the conversation.
- For angry customers or manager requests, use [escalate_to_human('reason')].
- Be concise. Solve in as few steps as possible.

Respond with ONLY your next tool call in brackets. Nothing else."""


def run_baseline():
    """Run the OpenAI API baseline against all 15 scenarios."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        print("       export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    env_client = SupportEnvClient(base_url=ENV_URL)

    print("\n" + "=" * 75)
    print(f"OPENENV CSA — OpenAI API BASELINE ({MODEL})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75 + "\n")

    results = {}

    for task_id in SCENARIOS:
        try:
            # Reset environment
            env_result = env_client.reset(task_id=task_id)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Add the customer's initial message
            customer_msg = ""
            for m in env_result.messages:
                if m["role"] == "customer":
                    customer_msg = m["content"]
            messages.append({"role": "user", "content": f"Customer: {customer_msg}"})

            grader_score = 0.0
            turn_count = 0

            for turn in range(MAX_TURNS):
                if env_result.done:
                    break

                # Call OpenAI API
                response = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.0  # Deterministic for reproducibility
                )

                agent_action = response.choices[0].message.content.strip()
                messages.append({"role": "assistant", "content": agent_action})
                turn_count += 1

                # Step the environment
                env_result = env_client.step(SupportAction(message=agent_action))

                # Get feedback and add to conversation
                feedback = ""
                for m in env_result.messages:
                    if m["role"] == "system":
                        feedback = m["content"]
                if feedback:
                    messages.append({"role": "user", "content": f"Environment: {feedback}"})

                if env_result.done:
                    grader_score = env_result.metadata.get("grader_score", 0.0)

            results[task_id] = {
                "score": grader_score,
                "turns": turn_count
            }

            status = "✅" if grader_score >= 0.8 else "⚠️" if grader_score > 0 else "❌"
            print(f"  {status} [{task_id:<22}] Score: {grader_score:.2f}  Turns: {turn_count}")

        except Exception as e:
            print(f"  ❌ [{task_id:<22}] ERROR: {e}")
            results[task_id] = {"score": 0.0, "turns": 0}

    # Summary
    scores = [r["score"] for r in results.values()]
    avg = sum(scores) / len(scores) if scores else 0.0

    easy_scores = [results[t]["score"] for t in SCENARIOS[:5]]
    medium_scores = [results[t]["score"] for t in SCENARIOS[5:10]]
    hard_scores = [results[t]["score"] for t in SCENARIOS[10:]]

    print("\n" + "=" * 75)
    print(f"  OVERALL:  {avg*100:.1f}%")
    print(f"  Easy:     {sum(easy_scores)/len(easy_scores)*100:.1f}%")
    print(f"  Medium:   {sum(medium_scores)/len(medium_scores)*100:.1f}%")
    print(f"  Hard:     {sum(hard_scores)/len(hard_scores)*100:.1f}%")
    print("=" * 75)

    # Save to CSV
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/baseline_openai_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Score", "Turns", "Model", "Timestamp"])
        ts = datetime.now().isoformat()
        for tid, data in results.items():
            writer.writerow([tid, data["score"], data["turns"], MODEL, ts])
        writer.writerow(["GLOBAL_AVERAGE", avg, "", MODEL, ts])
    print(f"\nResults saved to {csv_path}")

    return results, avg


if __name__ == "__main__":
    run_baseline()
