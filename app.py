import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import requests
import pandas as pd

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_env.client import SupportEnvClient, Action
from training.inference import run_benchmark
from training.config import METRICS_FILE, ENV_SERVER_URL
from agents.orchestrator import Orchestrator

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Scenario → Sample Messages ────────────────────────────────────────────
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

TOOL_CATALOG_MD = """
### 🛠️ Agent Tool Catalog

**📦 Order Agent Tools**
- `get_order(id)` — Order details
- `cancel_order(id)` — Cancel pending orders
- `validate_coupon(code)` — Apply discounts
- `reset_password(email)` — Account recovery

**🚚 Logistics Agent Tools**
- `track_shipment(id)` — Live tracking
- `update_address(id, addr)` — Change destination
- `check_delivery_slot(id)` — Available slots
- `reschedule_delivery(id, slot)` — Reschedule
- `investigate_missing(id)` — Missing item case

**💰 Finance Agent Tools**
- `validate_return(id)` — Return eligibility
- `ask_proof(id)` — Request damage evidence
- `create_return_request(id)` — Create return
- `initiate_refund(id)` — Reverse payment

**👨‍💼 Supervisor Tools**
- `escalate_to_human(reason)` — Human handover
- `respond(msg)` — Final customer response
"""

RULES_MD = """
**Rule 1 — Multi-Agent Pipeline**
Every request flows: 🧭 Router → Specialist → 👨‍💼 Supervisor

**Rule 2 — Tool-Call Format**
Agents use brackets: `[tool_name('param')]`

**Rule 3 — Turn Limit**
Max **6 turns** per interaction.

**Rule 4 — Escalation**
Angry/security issues bypass specialists → Supervisor direct.

**Rule 5 — Restricted Tools**
Each specialist can ONLY use its own tool set.
"""

# ── Load Model & Create Multi-Agent System ─────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Initializing Multi-Agent Orchestrator...")
orchestrator = Orchestrator(model, tokenizer, DEVICE)
client = SupportEnvClient(base_url=ENV_SERVER_URL)

# ── Helpers ────────────────────────────────────────────────────────────────
def wait_for_server(url):
    try: return requests.get(f"{url}/health").status_code == 200
    except: return False

def get_leaderboard_data():
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        chart_df = df[df["Scenario"] != "GLOBAL_AVERAGE"].copy()
        avg_row = df[df["Scenario"] == "GLOBAL_AVERAGE"]
        avg_score = avg_row["Score"].values[0] if not avg_row.empty else 0.0
        return chart_df, chart_df, f"### Current Master Table Accuracy: **{avg_score*100:.1f}%**"
    empty = pd.DataFrame(columns=["Scenario", "Score"])
    return empty, empty, "### No benchmark data yet — click **Run Full Benchmark** to start."

def trigger_benchmark():
    run_benchmark(save_to_csv=True)
    return get_leaderboard_data()

def update_samples(task_id):
    samples = SAMPLE_PROMPTS.get(task_id, ["Hello"])
    return gr.Dropdown(choices=samples, value=samples[0])

def chat_step(message, history, task_id):
    """Process one turn through the multi-agent pipeline."""
    if not message or not message.strip():
        return history, "", ""
    if not wait_for_server(ENV_SERVER_URL):
        history.append([message, "⚠️ Environment server is not responding."])
        return history, "", "⚠️ Server offline"

    # Reset environment on first message
    if len(history) == 0:
        client.reset(task_id=task_id)

    # Build history text for context
    history_text = "\n".join([f"Customer: {h[0]}\nAgent: {h[1]}" for h in history]) if history else ""

    # Get observation from the environment (use message as a step to see what happens)
    # But first, run it through the multi-agent orchestrator to decide the action
    action, trace = orchestrator.process(
        customer_message=message,
        observation_text=f"Customer says: {message}",
        task_id=task_id,
        history_text=history_text
    )

    # Execute the action in the environment
    res = client.step(Action(message=action))
    obs_text = ""
    if res.observation and res.observation.messages:
        last_msg = res.observation.messages[-1]
        obs_text = last_msg.content

    # Build the display response
    agent_flow = trace.flow()
    display_response = f"**{agent_flow}**\n\n"

    if obs_text:
        display_response += f"📡 Environment: {obs_text}\n\n"
    display_response += f"🤖 Action: `{action}`"

    if res.done:
        score = res.info.get("grader_score", 0.0)
        display_response += f"\n\n✅ **Task Complete** — Grader Score: **{score:.2f}**"

    history.append([message, display_response])
    trace_md = trace.summary()

    return history, "", trace_md


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), title="OpenEnv CSA Dashboard") as demo:
    gr.Markdown("# 🤖 OpenEnv CSA — Multi-Agent Dashboard")
    gr.Markdown("A multi-agent RL system: **Router → Specialist → Supervisor** pipeline for e-commerce support.")

    with gr.Row():
        # ── LEFT SIDEBAR ──
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Controls")
            task_selector = gr.Dropdown(choices=SCENARIOS, label="1. Active Scenario", value="easy_status")
            reset_btn = gr.Button("♻️ Reset Environment", variant="secondary")

            gr.Markdown("---")
            sample_selector = gr.Dropdown(choices=SAMPLE_PROMPTS["easy_status"], label="2. Sample Prompts", value=SAMPLE_PROMPTS["easy_status"][0])
            use_sample_btn = gr.Button("✨ Apply Sample", variant="secondary")

            with gr.Accordion("📜 Rules of the Agent", open=False):
                gr.Markdown(RULES_MD)
            with gr.Accordion("🛠️ Tool Catalog", open=False):
                gr.Markdown(TOOL_CATALOG_MD)

        # ── MAIN PANEL ──
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("💬 Agent Conversation"):
                    chatbot = gr.Chatbot(label="Multi-Agent Chat", height=400)
                    with gr.Row():
                        msg_input = gr.Textbox(placeholder="Type message…", label="Your Input", scale=4)
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear Chat")

                    gr.Markdown("### 🔍 Agent Trace")
                    trace_display = gr.Markdown("_Send a message to see the agent pipeline in action._")

                with gr.Tab("📊 Performance Metrics"):
                    accuracy_text = gr.Markdown("### Loading stats…")
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Stats")
                        run_btn = gr.Button("🚀 Run Full Benchmark", variant="primary")
                    plot = gr.BarPlot(x="Scenario", y="Score", title="Master Table Success Rate", y_lim=[0, 1], height=400, color="Scenario")
                    leaderboard_table = gr.DataFrame()

    # ── Events ──
    task_selector.change(update_samples, inputs=[task_selector], outputs=[sample_selector])
    use_sample_btn.click(lambda s: s, inputs=[sample_selector], outputs=[msg_input])
    submit_btn.click(chat_step, [msg_input, chatbot, task_selector], [chatbot, msg_input, trace_display])
    msg_input.submit(chat_step, [msg_input, chatbot, task_selector], [chatbot, msg_input, trace_display])
    reset_btn.click(lambda: ([], "", "_Send a message to see the agent pipeline._"), outputs=[chatbot, msg_input, trace_display])
    clear_btn.click(lambda: ([], "", "_Send a message to see the agent pipeline._"), outputs=[chatbot, msg_input, trace_display])
    refresh_btn.click(get_leaderboard_data, outputs=[plot, leaderboard_table, accuracy_text])
    run_btn.click(trigger_benchmark, outputs=[plot, leaderboard_table, accuracy_text])
    demo.load(get_leaderboard_data, outputs=[plot, leaderboard_table, accuracy_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
