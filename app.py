import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import time
import requests
import pandas as pd
from typing import List, Dict

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_env.client import SupportEnvClient, Action
from training.inference import run_benchmark
from training.config import METRICS_FILE, ENV_SERVER_URL

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scenarios from OpenEnv CSA
SCENARIOS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account",
    "medium_delay", "easy_cancel", "medium_address", "medium_reschedule", "medium_return",
    "medium_double_charge", "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
]

# Load Model and Tokenizer
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

client = SupportEnvClient(base_url=ENV_SERVER_URL)

def wait_for_server(url, timeout=5):
    try:
        response = requests.get(f"{url}/health")
        return response.status_code == 200
    except:
        return False

def get_leaderboard_data():
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        # Filter out the global average for the bar chart
        chart_df = df[df["Scenario"] != "GLOBAL_AVERAGE"]
        avg_row = df[df["Scenario"] == "GLOBAL_AVERAGE"]
        avg_score = avg_row["Score"].values[0] if not avg_row.empty else 0.0
        return chart_df, f"Current Accuracy: {avg_score*100:.1f}%"
    return pd.DataFrame(columns=["Scenario", "Score"]), "No benchmark data found. Run a benchmark to start tracking."

def trigger_benchmark():
    run_benchmark(save_to_csv=True)
    return get_leaderboard_data()

def predict(message, history, task_id):
    if not wait_for_server(ENV_SERVER_URL):
        return "Error: Environment server is not responding. Please check logs."

    # If history is empty, it's a new session
    if not history:
        res = client.reset(task_id=task_id)
        obs = res.observation
    else:
        res = client.step(Action(message=message))
        obs = res.observation
    
    prompt = f"System: You are a Customer Support Agent. Use tools or respond to the user.\nScenario: {task_id}\nObservation: {obs}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    return response

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 OpenEnv CSA: Customer Support Agent")
    
    with gr.Tabs():
        with gr.Tab("💬 Agent Chat"):
            with gr.Row():
                task_selector = gr.Dropdown(choices=SCENARIOS, label="Select Scenario", value="easy_status")
                reset_btn = gr.Button("Reset Environment")

            chat_ui = gr.ChatInterface(
                predict,
                additional_inputs=[task_selector],
                description="Type a tool call like `[get_order('ORD-101')]` or a text response.",
            )

        with gr.Tab("📊 Performance Leaderboard"):
            gr.Markdown("### Agent Master Table Performance")
            accuracy_text = gr.Markdown("Loading metrics...")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Stats")
                run_btn = gr.Button("🚀 Run Full Benchmark", variant="primary")
            
            plot = gr.BarPlot(
                value=None,
                x="Scenario",
                y="Score",
                title="Scenario Accuracy (0.0 - 1.0)",
                vertical=False,
                y_lim=[0, 1],
                width=800,
                height=400
            )

            leaderboard_table = gr.DataFrame(label="Latest Benchmarking Results")

            def update_ui():
                df, text = get_leaderboard_data()
                return df, df, text

            refresh_btn.click(update_ui, outputs=[plot, leaderboard_table, accuracy_text])
            run_btn.click(trigger_benchmark, outputs=[plot, leaderboard_table, accuracy_text])
            
            # Load initial data
            demo.load(update_ui, outputs=[plot, leaderboard_table, accuracy_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
