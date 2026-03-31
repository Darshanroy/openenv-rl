import sys
import os
import csv
from datetime import datetime
import torch

# Ensure imports work regardless of local execution path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env.client import SupportEnvClient, Action
from training.config import ENV_SERVER_URL, METRICS_FILE, LOG_DIR
from agents.orchestrator import Orchestrator

# 3 conversational variations per scenario to prove robust LLM behavior
EVAL_PROMPTS = {
    "easy_status": [
        "Where is my order ORD-101?",
        "Hey where is my order ORD-101?? I ordered it ages ago.",
        "Tracking for ORD-101, please."
    ],
    "easy_payment_fail": [
        "My payment for ORD-1414 failed.",
        "my card got declined for ORD-1414 but I have money??",
        "ORD-1414 payment error, what's wrong."
    ],
    "easy_coupon": [
        "I have a coupon SAVE10 but it's not working.",
        "Apply SAVE10 to my cart please, it keeps saying invalid.",
        "Code SAVE10 broke for me."
    ],
    "easy_account": [
        "I forgot my password for meera.reddy@example.com.",
        "can't login to meera.reddy@example.com pls reset",
        "lockout on meera.reddy@example.com"
    ],
    "medium_delay": [
        "My order ORD-909 is late.",
        "My order ORD-909 is like a week late, what is going on??",
        "Where the heck is ORD-909?"
    ],
    "easy_cancel": [
        "Cancel my order ORD-505 immediately.",
        "Cancel my order ORD-505 immediately, found it cheaper somewhere else.",
        "mistake order ORD-505 cancel pls."
    ],
    "medium_address": [
        "Change my address for ORD-1919 to '789 New Street'.",
        "Oops I put the wrong address for ORD-1919. Change it to '789 New Street' pls.",
        "moved recently, change address on ORD-1919 to 789 New Street."
    ],
    "medium_reschedule": [
        "Can we reschedule ORD-2323?",
        "I'm out of town, can we reschedule ORD-2323?",
        "reschedule ORD-2323"
    ],
    "medium_return": [
        "I want to return ORD-2020.",
        "The items in ORD-2020 don't fit, how do I return them?",
        "want to return ORD-2020, didn't like it."
    ],
    "medium_double_charge": [
        "I was charged twice for ORD-1515.",
        "UMM why was I charged TWICE for ORD-1515??? Fix this now.",
        "double charge bug on ORD-1515."
    ],
    "hard_refund": [
        "I need a full refund for ORD-2121.",
        "I need a full refund for ORD-2121. The quality is terrible.",
        "processing refund for ORD-2121."
    ],
    "hard_damaged": [
        "My order ORD-2222 is damaged.",
        "ORD-2222 arrived completely shattered in pieces! Unbelievable!!!",
        "item broken in ORD-2222."
    ],
    "hard_missing": [
        "My order ORD-1313 shows as delivered but it's not here.",
        "Tracking for ORD-1313 shows as 'Delivered' but there is literally nothing on my porch.",
        "ORD-1313 says delivered, it's NOT here."
    ],
    "hard_angry": [
        "I'm EXTREMELY angry! ORD-909 is still missing!",
        "I AM EXTREMELY FURIOUS! ORD-909 IS STILL MISSING AND NOBODY IS HELPING ME!!!",
        "Worst experience ever. Where is ORD-909, you guys are a scam."
    ],
    "hard_escalation": [
        "I want to speak to your manager about ORD-1414.",
        "I want to speak to your manager about ORD-1414 right away.",
        "Connect me to a human supervisor now, I'm done with bots."
    ]
}

def run_benchmark(orchestrator=None, save_to_csv=True):
    """
    Automated LLM Evaluation using the Multi-Agent Orchestrator.
    Tests 3 varied inputs per scenario to prove robust language understanding over hardcoded rules.
    """
    client = SupportEnvClient(base_url=ENV_SERVER_URL)
    
    if orchestrator is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        MODEL_NAME = "Qwen/Qwen2.5-1.5B"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
        )
        orchestrator = Orchestrator(model, tokenizer, DEVICE)

    print("\n" + "="*80)
    print(f"OPENENV CSA LIVE AGENT EVALUATION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    results = {}
    MAX_TURNS = 6

    for tid, variations in EVAL_PROMPTS.items():
        print(f"[{tid.upper()}] evaluating {len(variations)} linguistic variations...")
        variation_scores = []
        
        for idx, customer_msg in enumerate(variations):
            try:
                res = client.reset(task_id=tid)
                history_lines = []
                grader_score = 0.0
                obs_text = f"Customer says: {customer_msg}"
                
                # Run conversation loop
                for turn in range(MAX_TURNS):
                    # Multi-agent inference
                    action, trace = orchestrator.process(
                        customer_message=customer_msg,
                        observation_text=obs_text,
                        task_id=tid,
                        history_text="\n".join(history_lines)
                    )
                    
                    history_lines.append(f"Customer: {customer_msg}" if turn == 0 else f"Agent: {action}")
                    
                    # Environment step
                    res = client.step(Action(message=action))
                    if res.observation and res.observation.messages:
                        obs_text = res.observation.messages[-1].content
                        history_lines.append(f"Environment: {obs_text}")
                    
                    if res.done:
                        grader_score = res.info.get("grader_score", 0.0)
                        break

                variation_scores.append(grader_score)
                print(f"   -> Var {idx+1} Score: {grader_score:.2f}")

            except Exception as e:
                print(f"   -> Var {idx+1} Error: {e}")
                variation_scores.append(0.0)
        
        # Average across the 3 variations for this task
        avg_task_score = sum(variation_scores) / len(variation_scores)
        results[tid] = avg_task_score
        print(f"   => Average Score: {avg_task_score:.2f}\n")

    avg_score = sum(results.values()) / len(results)
    print("="*80)
    print(f"GLOBAL PERFORMANCE (LLM Multi-Agent): {avg_score*100:.1f}% ACCURACY")
    print("="*80 + "\n")

    if save_to_csv:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        
        with open(METRICS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario", "Score", "Timestamp"])
            ts = datetime.now().isoformat()
            for tid, score in results.items():
                writer.writerow([tid, score, ts])
            writer.writerow(["GLOBAL_AVERAGE", avg_score, ts])
        print(f"Live benchmark saved to {METRICS_FILE}")

    return results, avg_score

if __name__ == "__main__":
    run_benchmark()
