import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import requests
import pandas as pd
import time

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_env.client import SupportEnvClient, Action
from training.inference import run_benchmark
from training.config import METRICS_FILE, ENV_SERVER_URL
from agents.orchestrator import Orchestrator

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_PROMPTS = {
    "easy_status": ["Where is my order ORD-101?", "When will ORD-101 arrive?", "Status of my latest order?"],
    "easy_payment_fail": ["My payment for ORD-1414 failed.", "Why did my transaction for ORD-1414 not go through?"],
    "easy_coupon": ["I have a coupon SAVE10 but it's not working.", "Apply SAVE10 to my cart please."],
    "easy_account": ["I forgot my password for meera.reddy@example.com.", "Reset my account password."],
    "medium_delay": ["My order ORD-909 is late.", "Check delay on ORD-909."],
    "easy_cancel": ["Cancel my order ORD-505 immediately.", "I don't want ORD-505 anymore."],
    "medium_address": ["Change my address for ORD-1919 to '789 New Street'.", "Wrong address for ORD-1919."],
    "medium_reschedule": ["Can we reschedule ORD-2323?", "Change delivery date for ORD-2323."],
    "medium_return": ["I want to return ORD-2020.", "Requesting return for ORD-2020."],
    "medium_double_charge": ["I was charged twice for ORD-1515.", "Refund the second charge on ORD-1515."],
    "hard_refund": ["I need a full refund for ORD-2121.", "I want to refund ORD-2121."],
    "hard_damaged": ["My order ORD-2222 is damaged.", "The box for ORD-2222 was crushed."],
    "hard_missing": ["My order ORD-1313 shows as delivered but it's not here.", "Missing items from ORD-1313."],
    "hard_angry": ["I'm EXTREMELY angry! ORD-909 is still missing!", "This is terrible service for ORD-909!"],
    "hard_escalation": ["I want to speak to your manager about ORD-1414.", "Connect me to a supervisor."]
}
SCENARIOS = list(SAMPLE_PROMPTS.keys())

# ── Setup Caching & State ────────────────────────────────────────────────
st.set_page_config(page_title="OpenEnv CSA Dashboard", page_icon="🤖", layout="wide")

@st.cache_resource(show_spinner="Loading Model weights to GPU... This takes a minute.")
def load_system():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    orch = Orchestrator(model, tokenizer, DEVICE)
    client = SupportEnvClient(base_url=ENV_SERVER_URL)
    return orch, client

def wait_for_server(url):
    try: return requests.get(f"{url}/health").status_code == 200
    except: return False

orchestrator, client = load_system()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "scores" not in st.session_state:
    st.session_state.scores = None

def reset_chat(task_id):
    st.session_state.messages = []
    st.session_state.scores = None
    if wait_for_server(ENV_SERVER_URL):
        try:
            client.reset(task_id=task_id)
        except Exception as e:
            st.error(f"Environment reset failed: {e}")

# ── Sidebar Configuration ──
with st.sidebar:
    st.title("⚙️ Mission Control")
    st.markdown("Guide the multi-agent system through 15 e-commerce scenarios.")
    
    selected_task = st.selectbox("1. Select Task Tier", options=SCENARIOS, 
                                 help="Choose the difficulty and scenario you want to test.")
    
    st.markdown("### Suggested Openers")
    for prompt in SAMPLE_PROMPTS.get(selected_task, []):
        st.code(prompt, language=None)
        
    if st.button("♻️ Reset Environment & Clear Chat", use_container_width=True):
        reset_chat(selected_task)
        st.rerun()

    with st.expander("📖 Agent Protocol Rules"):
        st.markdown("""
        **1. Strict Routing**: Requests always pass from Router → Specialist → Supervisor.
        **2. Tool Constraints**: Specialists can only use their restricted toolset (e.g. Finance cannot cancel orders).
        **3. Supervisor Override**: Abusive inputs or manager requests bypass specialists completely.
        **4. Turn Limits**: The environment caps episodes at 6 turns.
        """)

# ── Main Content Area ──
st.title("🤖 OpenEnv Multi-Agent Dashboard")
st.markdown("Watch the pipeline dynamically route intents, execute tools, and finalize responses.")

tab_chat, tab_metrics = st.tabs(["💬 Live Testing & Flow", "📊 Benchmark Leaderboard"])

with tab_chat:
    
    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if "trace" in msg and msg["trace"]:
                with st.expander(f"🔍 Agent Pipeline Activity: {msg['agent_flow']}"):
                    st.markdown(msg["trace"])
            if "env" in msg and msg["env"]:
                st.info(f"📡 Environment Database:\n\n{msg['env']}")

    # Handle incoming messages
    if user_input := st.chat_input("Type your message here..."):
        if not wait_for_server(ENV_SERVER_URL):
            st.error("⚠️ Environment API server is offline. Please start it using uvicorn.")
            st.stop()
            
        # First message handler (if user didn't click reset)
        if len(st.session_state.messages) == 0:
            reset_chat(selected_task)
        
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Agent Processing Turn
        with st.chat_message("assistant", avatar="🤖"):
            with st.status(f"Pipeline active. Routing intent...", expanded=True) as status:
                
                # Build context
                history_text = "\n".join([f"{'Customer' if x['role']=='user' else 'Agent'}: {x['content']}" 
                                          for x in st.session_state.messages])
                
                # multi-agent processing
                status.update(label="Running Router & Specialist...")
                action, trace = orchestrator.process(
                    customer_message=user_input,
                    observation_text=f"Customer says: {user_input}",
                    task_id=selected_task,
                    history_text=history_text
                )
                
                # Env step
                status.update(label=f"Executing Tool: {action}")
                res = client.step(Action(message=action))
                obs_text = ""
                if res.observation and res.observation.messages:
                    last_msg = res.observation.messages[-1]
                    obs_text = last_msg.content

                status.update(label="Supervisor generated final output", state="complete")

            # Result Rendering
            flow_line = trace.flow()
            st.markdown(f"**Final Agent Output:**\n`{action}`")
            
            if obs_text:
                st.info(f"📡 Environment Data Sync:\n\n{obs_text}")
                
            if res.done:
                score = res.info.get("grader_score", 0.0)
                st.success(f"✅ **Task Ended** — Final Grader Score: **{score:.2f}**")
            
            # Save to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Final Action:** `{action}`", 
                "trace": trace.summary(),
                "agent_flow": flow_line,
                "env": obs_text if obs_text else None
            })
            st.rerun()

with tab_metrics:
    st.markdown("### 📈 Reinforcement Learning Benchmarks")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Run Full 15-Task Eval", use_container_width=True):
            with st.spinner("Running Hardcoded Assessment..."):
                run_benchmark(save_to_csv=True)
                st.rerun()
                
    with col2:
        if os.path.exists(METRICS_FILE):
            df = pd.read_csv(METRICS_FILE)
            avg_row = df[df["Scenario"] == "GLOBAL_AVERAGE"]
            if not avg_row.empty:
                st.metric(label="Global Average Accuracy", value=f"{avg_row['Score'].values[0]*100:.1f}%")
            
            chart_df = df[df["Scenario"] != "GLOBAL_AVERAGE"].copy()
            if not chart_df.empty:
                st.bar_chart(data=chart_df, x="Scenario", y="Score", use_container_width=True)
                st.dataframe(chart_df, use_container_width=True)
        else:
            st.info("No leaderboard data found. Click 'Run Full 15-Task Eval' to generate the baseline metrics.")
