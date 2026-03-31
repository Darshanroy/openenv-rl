import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import time
import requests
from typing import List, Dict

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_env.client import SupportEnvClient, Action

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ENV_URL = "http://127.0.0.1:8000"

# Scenarios from OpenEnv CSA
SCENARIOS = [
    "easy_status", "easy_payment_fail", "easy_coupon", "easy_account",
    "medium_delay", "easy_cancel", "medium_address", "medium_reschedule", "medium_return",
    "medium_double_charge", "hard_refund", "hard_damaged", "hard_missing", "hard_angry", "hard_escalation"
]

# Load Model and Tokenizer
print(f"Loading model {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

client = SupportEnvClient(base_url=ENV_URL)

def wait_for_server(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("Environment server is up!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def predict(message, history, task_id):
    if not wait_for_server(ENV_URL, timeout=5):
        return "Error: Environment server is not responding. Please check logs."

    # If history is empty, it's a new session
    if not history:
        res = client.reset(task_id=task_id)
        obs = res.observation
    else:
        # Step with the message (assuming it's a tool call or response)
        res = client.step(Action(message=message))
        obs = res.observation
    
    # Simple model inference (this is a placeholder for the actual RL agent logic)
    # In a real deployment, you'd use the trained model weights.
    prompt = f"System: You are a Customer Support Agent. Use tools or respond to the user.\n"
    prompt += f"Scenario: {task_id}\n"
    prompt += f"Observation: {obs}\n"
    prompt += "Assistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    
    # If the model suggested a tool call, the user can see it.
    # In a fully autonomous mode, we would parse and execute it here.
    return response

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 OpenEnv CSA: Customer Support Agent")
    gr.Markdown("Interact with our RL-trained agent in various simulated ecommerce scenarios.")
    
    with gr.Row():
        task_selector = gr.Dropdown(choices=SCENARIOS, label="Select Scenario", value="easy_status")
        reset_btn = gr.Button("Reset Environment")

    chatbot = gr.ChatInterface(
        predict,
        additional_inputs=[task_selector],
        description="Type a tool call like `[get_order('ORD-101')]` or a text response.",
        examples=[["[get_order('ORD-101')]"], ["I'm sorry for the delay."]]
    )

    reset_btn.click(lambda: [], outputs=None) # Simple UI reset trigger logic would go here

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
