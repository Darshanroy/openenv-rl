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

# Scenario Mapping
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

# Tool Catalog Data
TOOL_CATALOG = """
### 🛠️ Agent Tool Catalog
The agent uses these APIs to interact with the database:

**Order Management**
- `get_order(order_id)`: Fetches details like items, price, and status.
- `cancel_order(order_id)`: Cancels pending orders.

**Logistics & Delivery**
- `track_shipment(order_id)`: Live tracking data.
- `update_address(order_id, addr)`: Changes destination.
- `reschedule_delivery(order_id, slot)`: Changes delivery time.

**Returns & Refunds**
- `validate_return(order_id)`: Eligibility check.
- `ask_proof(order_id)`: Requests damage photo links.
- `initiate_refund(order_id)`: Reverses payment.

**Support**
- `validate_coupon(code)`: Applies discounts.
- `reset_password(email)`: Account recovery.
- `escalate_to_human(issue)`: Human handover.
"""

# Theme Setup
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    block_border_width="1px",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

# Load Model/Client
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
client = SupportEnvClient(base_url=ENV_SERVER_URL)

def wait_for_server(url):
    try:
        return requests.get(f"{url}/health").status_code == 200
    except:
        return False

def get_leaderboard_data():
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        chart_df = df[df["Scenario"] != "GLOBAL_AVERAGE"]
        avg_row = df[df["Scenario"] == "GLOBAL_AVERAGE"]
        avg_score = avg_row["Score"].values[0] if not avg_row.empty else 0.0
        return chart_df, f"Current Master Table Accuracy: {avg_score*100:.1f}%"
    return pd.DataFrame(columns=["Scenario", "Score"]), "No benchmark data found. Run a full benchmark to start tracking."

def trigger_benchmark():
    run_benchmark(save_to_csv=True)
    return get_leaderboard_data()

def update_samples(task_id):
    samples = SAMPLE_PROMPTS.get(task_id, ["Hello"])
    return gr.update(choices=samples, value=samples[0])

def chat_step(message, history, task_id):
    if not wait_for_server(ENV_SERVER_URL):
        return history + [[message, "Error: Environment server is not responding. Please check logs."]], ""

    if len(history) == 0:
        res = client.reset(task_id=task_id)
        obs = res.observation
    else:
        res = client.step(Action(message=message))
        obs = res.observation
    
    prompt = f"System: You are a Customer Support Agent. Use tools or respond to the user.\nScenario: {task_id}\nObservation: {obs}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    return history + [[message, response]], ""

# --- UI APP ---
with gr.Blocks(theme=theme, title="OpenEnv CSA Dashboard") as demo:
    gr.Markdown("# 🤖 OpenEnv CSA Dashboard")
    gr.Markdown("Interact with our RL-trained Customer Support Agent and track performance stats.")
    
    with gr.Row():
        # --- LEFT SIDEBAR ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Controls")
                task_selector = gr.Dropdown(choices=SCENARIOS, label="1. Active Scenario", value="easy_status")
                reset_btn = gr.Button("♻️ Reset Environment", variant="secondary")
                
                gr.Markdown("---")
                sample_selector = gr.Dropdown(
                    choices=SAMPLE_PROMPTS["easy_status"], 
                    label="2. Conversational Samples", 
                    value=SAMPLE_PROMPTS["easy_status"][0]
                )
                use_sample_btn = gr.Button("✨ Apply Sample to Chat", variant="secondary")

            with gr.Accordion("📜 Rules of the Agent", open=False):
                gr.Markdown("""
                **Rule 1: Tool Call Format**
                The agent communicates with the database using brackets: `[tool_name('parameter')]`.
                
                **Rule 2: Turn Limit**
                Every interaction has a **6-turn limit**. If the agent hasn't resolved the issue by then, it's marked as failed.
                
                **Rule 3: Handover**
                For critical issues (Angry/Security), the agent is trained to use `escalate_to_human`.
                """)

            with gr.Accordion("🛠️ Tool Catalog", open=False):
                gr.Markdown(TOOL_CATALOG)

        # --- MAIN PANEL ---
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("💬 Agent Conversation"):
                    chatbot = gr.Chatbot(label="Simulated Chat Session", height=450, bubble_full_width=False)
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message here...", 
                            label="Your Input", 
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    gr.ClearButton([chatbot, msg_input], value="🗑️ Clear Chat")

                with gr.Tab("📊 Performance Metrics"):
                    accuracy_text = gr.Markdown("Loading Master Table stats...")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Stats")
                        run_btn = gr.Button("🚀 Run Full Benchmark", variant="primary")
                    
                    plot = gr.BarPlot(
                        value=None, x="Scenario", y="Score",
                        title="Scenario Success Rate (0.0 - 1.0)",
                        vertical=False, y_lim=[0, 1],
                        width=None, height=400, color="Scenario",
                        tooltip=["Scenario", "Score"]
                    )
                    
                    gr.Markdown("### 📋 Latest Detailed Report")
                    leaderboard_table = gr.DataFrame()

    # --- EVENTS ---
    task_selector.change(update_samples, inputs=[task_selector], outputs=[sample_selector])
    use_sample_btn.click(lambda s: s, inputs=[sample_selector], outputs=[msg_input])
    
    # Combined Chat Events
    submit_btn.click(chat_step, [msg_input, chatbot, task_selector], [chatbot, msg_input])
    msg_input.submit(chat_step, [msg_input, chatbot, task_selector], [chatbot, msg_input])
    reset_btn.click(lambda: [], None, [chatbot])

    # Leaderboard Events
    def update_ui():
        df, text = get_leaderboard_data()
        return df, df, text

    refresh_btn.click(update_ui, outputs=[plot, leaderboard_table, accuracy_text])
    run_btn.click(trigger_benchmark, outputs=[plot, leaderboard_table, accuracy_text])
    
    # Initialization
    demo.load(update_ui, outputs=[plot, leaderboard_table, accuracy_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
