---
title: OpenEnv-CSA-RL
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🤖 OpenEnv Autonomous Customer Support Agent (CSA)

Welcome to the **OpenEnv CSA** project—a production-grade, multi-agent reinforcement learning environment and agent system. This repository features a decoupled architecture designed for high-performance evaluation and training.

## 📂 Project Structure

```text
.
├── agents/             # Multi-agent reasoning logic (The 'Brain')
│   ├── orchestrator.py # High-level coordination (Router → Specialist → Supervisor)
│   ├── router.py       # Intent classification and urgency detection
│   ├── specialist.py   # Domain-specific tool usage (Order, Logistics, Finance)
│   └── supervisor.py   # Quality control, empathy, and escalation logic
├── server/             # Environment API (The 'World')
│   ├── app.py          # FastAPI application & session-based REST routes
│   ├── db.py           # Comprehensive E-commerce Mock Database (53KB)
│   ├── my_environment.py # Core SupportEnvironment (OpenEnv Spec)
│   └── tools.py        # 17 specialized support tools (Refunds, Tracking, etc.)
├── training/           # RL training utilities
│   ├── rewards.py      # 9-function weighted reward pipeline for GRPO/PPO
│   ├── support_tool_env.py # TRL environment_factory class
│   ├── verify_graders.py   # Grader compliance verification (strict (0,1) bounds)
│   └── inference.py    # Training-time benchmark runner
├── .env                # Local secrets and configuration (HF_TOKEN)
├── Dockerfile          # Production deployment specification (Hugging Face)
├── inference.py        # Main evaluation entry point (15-Task Suite)
├── openenv.yaml        # OpenEnv submission metadata and entry points
└── README.md           # This file
```

## 🏗️ System Architecture

This project follows a **Decoupled Architecture**, separating the "World" (Environment) from the "Brain" (Agent).

- **The World (Environment)**: A standalone FastAPI server hosted in a Docker container (HF Space). It manages the e-commerce database, tool execution, and rewards.
- **The Brain (Agent)**: A multi-agent orchestrator that runs locally and communicates with the environment via REST API.

```mermaid
graph LR
    A[Inference Script] --> B[Multi-Agent Orchestrator]
    subgraph Agents
        B --> C[Router Agent]
        B --> D[Specialist Agent]
        B --> E[Supervisor Agent]
    end
    Agents --> F[Environment API]
    F --> G[(E-Commerce DB)]
```

## 🎯 Scoring Protocol

All scores in this system are **strictly between 0 and 1** (never exactly `0.0` or `1.0`):

| Score Type | Range | Method | Purpose |
| :--- | :--- | :--- | :--- |
| **Step Reward** | `(0.01, 0.99)` | Sigmoid squashing | Per-turn RL signal |
| **Grader Score** | `(0.01, 0.99)` | Linear normalization | Final task completion metric |

### How Scoring Works

- **Step Rewards**: Each agent-environment interaction produces a raw reward (can be negative for penalties or large positive for bonuses). These are squashed through a sigmoid function `σ(r/5.0)` and mapped to `(0.01, 0.99)` to prevent gradient explosion during RL training.
- **Grader Scores**: When a task ends, the environment computes a weighted sum of which correct tools the agent used. This raw `[0, 1]` score is linearly mapped to `(0.01, 0.99)` via `(score × 0.98) + 0.01`.
- **Protocol Enforcement**: For certain hard tasks (e.g., `hard_damaged`), the grader requires prerequisite tools (e.g., `ask_proof` before `initiate_refund`). Skipping prerequisites blocks the reward for the main tool.

## 🚀 Key Features

- **Multi-Agent Reasoning**: A specialized pipeline (Router → Specialist → Supervisor) that prevents "thought loops" and ensures high-quality tool calls.
- **15-Task Evaluation Suite**: 15 distinct e-commerce scenarios across **Easy, Medium, and Hard** difficulty tiers.
- **Autonomous Toolset**: 17 specialized tools for order tracking, refund validation, address changes, and more.
- **53KB Knowledge Base**: A massive, pre-populated database in `server/db.py` ensuring realistic scenarios.
- **Strict Score Normalization**: All rewards and scores are strictly in `(0, 1)` for RL training stability.
- **Per-Tier Reporting**: Evaluation output includes Easy/Medium/Hard averages for granular performance analysis.
- **RLHF-Ready**: Built-in feedback logging for future Reinforcement Learning from Human Feedback.

## 🛠️ Quick Start

### 1. Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your Hugging Face credentials:
```env
HF_TOKEN=your_token_here
ENV_URL=https://darshankumarr03-openenv-csa-rl.hf.space
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 3. Run Evaluation
Execute the multi-agent inference script to run the full 15-task evaluation:
```bash
python inference.py
```

Expected output format:
```text
[START] task=easy_status env=CustomerSupport-v1 model=Qwen/Qwen2.5-7B-Instruct
[STEP]  step=1 action=[get_order('ORD-101')] reward=0.69 done=false error=null
[STEP]  step=2 action=[track_shipment('ORD-101')] reward=0.69 done=false error=null
[STEP]  step=3 action=[respond('Your order is on the way!')] reward=0.96 done=true error=null
[END]   task=easy_status score=0.99 steps=3

============================================================
FINAL EVALUATION SUMMARY
============================================================
  easy_status               |   0.99
  ...
------------------------------------------------------------
  Easy Avg                  | 0.XXXX  (5 tasks)
  Medium Avg                | 0.XXXX  (5 tasks)
  Hard Avg                  | 0.XXXX  (5 tasks)
------------------------------------------------------------
  OVERALL AVERAGE           | 0.XXXX  (15 tasks)
============================================================
```

## 🌐 Environment API Specification

The Hugging Face Space provides a session-based REST API for high-performance agent interaction and RL training.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Liveness probe (OpenEnv Validator) |
| `/session/reset` | `POST` | Initialize a session with `session_id` and `task_id` |
| `/session/step/{id}` | `POST` | Execute a tool call/action for a specific session |
| `/session/state/{id}` | `GET` | Retrieve the current full state of the environment |
| `/session/feedback/{id}`| `POST` | Log RLHF feedback (`thumbs_up`/`thumbs_down`) |

## ✅ Validation Status

This project is officially **OpenEnv-compliant** and passes 3/3 validator checks:
1. **Metadata**: Valid `openenv.yaml`.
2. **Environment**: Successfully instantiates the `server` package.
3. **API**: Clean health checks and step-by-step connectivity.

---
**Author**: Darshankumarr03  
**Version**: 2.2.0  
**License**: MIT
