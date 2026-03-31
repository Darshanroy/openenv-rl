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

# Scenario to Conversational Sample Mapping
SAMPLE_PROMPTS = {
    "easy_status": ["Where is my order ORD-101?", "When will ORD-101 arrive?", "Status of my latest order?"],
    "easy_payment_fail": ["My payment for ORD-1414 failed.", "Why did my transaction for ORD-1414 not go through?", "Help with payment failure."],
    "easy_coupon": ["I have a coupon SAVE10 but it's not working.", "Apply SAVE10 to my cart please.", "Validating SAVE10 coupon."],
    "easy_account": ["I forgot my password for meera.reddy@example.com.", "Reset my account password.", "Login issues with my meera email."],
    "medium_delay": ["My order ORD-909 is late.", "When will ORD-909 get here? It's delayed.", "Check delay on ORD-909."],
    "easy_cancel": ["Cancel my order ORD-505 immediately.", "I don't want ORD-505 anymore.", "Stop the delivery for ORD-505."],
    "medium_address": ["Change my address for ORD-1919 to '789 New Street'.", "Update delivery location for ORD-1919.", "Wrong address for ORD-1919."],
    "medium_reschedule": ["Can we reschedule ORD-2323?", "I won't be home for ORD-2323, change time.", "Change delivery date for ORD-2323."],
    "medium_return": ["I want to return ORD-2020.", "The items in ORD-2020 are wrong, I want a return.", "Requesting return for ORD-2020."],
    "medium_double_charge": ["I was charged twice for ORD-1515.", "Refund the second charge on ORD-1515.", "Double payment for ORD-1515."],
    "hard_refund": ["I need a full refund for ORD-2121.", "Give me my money back for ORD-2121.", "I want to refund ORD-2121."],
    "hard_damaged": ["My order ORD-2222 is damaged.", "ORD-2222 arrived broken in pieces.", "The box for ORD-2222 was crushed."],
    "hard_missing": ["My order ORD-1313 shows as delivered but it's not here.", "Missing items from ORD-1313.", "Where is ORD-1313? I checked everywhere."],
    "hard_angry": ["I'm EXTREMELY angry! ORD-909 is still missing!", "This is terrible service for ORD-909!", "Worst experience ever with ORD-909!"],
    "hard_escalation": ["I want to speak to your manager about ORD-1414.", "Escalate my case for ORD-1414.", "Connect me to a supervisor."]
}

SCENARIOS = list(SAMPLE_PROMPTS.keys())

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
        chart_df = df[df["Scenario"] != "GLOBAL_AVERAGE"]
        avg_row = df[df["Scenario"] == "GLOBAL_AVERAGE"]
        avg_score = avg_row["Score"].values[0] if not avg_row.empty else 0.0
        return chart_df, f"Current Accuracy: {avg_score*100:.1f}%"
    return pd.DataFrame(columns=["Scenario", "Score"]), "No benchmark data found. Run a benchmark to start tracking."

def trigger_benchmark():
    run_benchmark(save_to_csv=True)
    return get_leaderboard_data()

def update_samples(task_id):
    samples = SAMPLE_PROMPTS.get(task_id, ["Hello"])
    return gr.update(choices=samples, value=samples[0])

def predict(message, history, task_id):
    if not wait_for_server(ENV_SERVER_URL):
        return history + [[message, "Error: Environment server is not responding. Please check logs."]]

    # If it's the start, reset the environment
    is_reset = len(history) == 0
    if is_reset:
        res = client.reset(task_id=task_id)
        obs = res.observation
    else:
        res = client.step(Action(message=message))
        obs = res.observation
    
    prompt = f"System: You are a Customer Support Agent. Use tools or respond to the user.\nScenario: {task_id}\nObservation: {obs}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    
    return history + [[message, response]]

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 OpenEnv CSA: Customer Support Agent")
    
    with gr.Tabs():
        with gr.Tab("💬 Agent Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_selector = gr.Dropdown(choices=SCENARIOS, label="1. Select Scenario", value="easy_status")
                    reset_btn = gr.Button("Reset Environment")
                
                with gr.Column(scale=2):
                    sample_selector = gr.Dropdown(choices=SAMPLE_PROMPTS["easy_status"], label="2. View Real Conversational Samples", value=SAMPLE_PROMPTS["easy_status"][0])
                    use_sample_btn = gr.Button("Apply Sample to Chat", variant="secondary")

            chatbot = gr.Chatbot(label="Customer Conversation")
            msg_input = gr.Textbox(placeholder="Ask the agent something...", label="User Message")
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

            def chat_step(message, history, task_id):
                return predict(message, history, task_id), ""

            # Connect Components
            task_selector.change(update_samples, inputs=[task_selector], outputs=[sample_selector])
            use_sample_btn.click(lambda s: s, inputs=[sample_selector], outputs=[msg_input])
            submit_btn.click(chat_step, inputs=[msg_input, chatbot, task_selector], outputs=[chatbot, msg_input])
            msg_input.submit(chat_step, inputs=[msg_input, chatbot, task_selector], outputs=[chatbot, msg_input])
            reset_btn.click(lambda: None, None, [chatbot]) # Just clear visual UI for now, logic handles it.

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
