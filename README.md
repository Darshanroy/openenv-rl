---
title: OpenEnv CSA - Multi-Agent RL System
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv CSA — Multi-Agent Customer Support Environment

A complete, real-world **OpenEnv environment** where AI agents learn to resolve e-commerce customer support issues through the standard `step()` / `reset()` / `state()` API.

## Architecture

This system uses a **multi-agent pipeline** powered by a single shared RL model (`Qwen/Qwen2.5-1.5B`) with role-specific system prompts:

```
Customer Message
       ↓
🧭 Router Agent        (rule-based intent classifier)
       ↓
┌──────┼──────┐
📦 Order  🚚 Logistics  💰 Finance    (specialist agents)
└──────┼──────┘
       ↓
👨‍💼 Supervisor Agent   (review, respond, or escalate)
       ↓
   Final Response
```

| Agent | Role | Tools |
|:---|:---|:---|
| 🧭 Router | Classifies intent → dispatches | None (rule-based) |
| 📦 Order | Orders, cancellations, coupons, accounts | `get_order`, `cancel_order`, `validate_coupon`, `reset_password` |
| 🚚 Logistics | Shipping, tracking, address, rescheduling | `track_shipment`, `update_address`, `check_delivery_slot`, `reschedule_delivery`, `investigate_missing` |
| 💰 Finance | Returns, refunds, damage claims | `validate_return`, `ask_proof`, `create_return_request`, `initiate_refund` |
| 👨‍💼 Supervisor | Reviews output, handles escalation | `escalate_to_human`, `respond` |

## Environment Specification

### Action Space
The agent sends **text actions** formatted as tool calls in brackets:
```
[get_order('ORD-101')]
[track_shipment('ORD-101')]
[respond('Your order will arrive on April 2nd.')]
[escalate_to_human('Customer requests manager')]
```

### Observation Space
Each observation contains:
- `prompt`: System instructions with available tools.
- `messages`: A list of `Message` objects with `category` (CUSTOMER / AGENT / FEEDBACK / SYSTEM) and `content`.

### Reward Function
| Signal | Description | Value |
|:---|:---|:---|
| Step penalty | Every turn costs | -1.0 |
| Intent reward | Correct tool for the scenario | +2 to +12 |
| Resolution bonus | Finishing with `respond()` | +10.0 |
| Efficiency bonus | Solving within optimal steps | +3.0 |
| Syntax error | Malformed tool call | -4.0 |
| Unknown tool | Tool not in registry | -5.0 |

### Grader (0.0 – 1.0)
Each scenario has `grader_weights` mapping tools to partial credit. The final grader score is the weighted sum of tools successfully used.

## Tasks (15 Scenarios)

### Easy (5 tasks)
| ID | Customer Problem | Optimal Steps |
|:---|:---|:---|
| `easy_status` | "Where is my order ORD-101?" | 3 |
| `easy_payment_fail` | "My payment for ORD-1414 failed." | 2 |
| `easy_coupon` | "Coupon SAVE10 isn't working." | 2 |
| `easy_account` | "Forgot password for meera.reddy@example.com." | 2 |
| `easy_cancel` | "Cancel order ORD-505 immediately." | 2 |

### Medium (5 tasks)
| ID | Customer Problem | Optimal Steps |
|:---|:---|:---|
| `medium_delay` | "ORD-909 is a week late!" | 3 |
| `medium_address` | "Change address for ORD-1919." | 2 |
| `medium_reschedule` | "Reschedule delivery for ORD-2323." | 3 |
| `medium_return` | "Return items from ORD-2020." | 3 |
| `medium_double_charge` | "Charged twice for ORD-1515." | 2 |

### Hard (5 tasks)
| ID | Customer Problem | Optimal Steps |
|:---|:---|:---|
| `hard_refund` | "Refund ORD-2121 now!" | 3 |
| `hard_damaged` | "ORD-2222 arrived shattered!" | 4 |
| `hard_missing` | "ORD-1313 says delivered but not here." | 4 |
| `hard_angry` | "YOUR SERVICE IS PATHETIC!" | 3 |
| `hard_escalation` | "I want to talk to your manager." | 2 |

## Setup & Installation

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000

# 3. Run baseline benchmark
python training/inference.py

# 4. Start the Gradio UI
python app.py
```

### Docker
```bash
docker build -t openenv-csa .
docker run -p 7860:7860 openenv-csa
```

### Training (GRPO)
```bash
# Ensure environment server is running on port 8000
python training/train.py
```

## Evaluation & Benchmarks

### Live Multi-Agent Evaluator (`training/inference.py`)
This script instantiates the full Router → Specialist pipeline and dynamically tests the agents on **3 conversational variations for all 15 intents** (45 unique scenarios).
It proves the agents genuinely parse intent and generate tools organically, handling slang, missing contexts, and anger without hardcoded paths.

```bash
# Run locally (uses the local Qwen/Qwen2.5-1.5B model)
python training/inference.py
```

| Tier | Expected Live Accuracy |
|:---|:---|
| Easy | ~80-100% |
| Medium | ~60-90% |
| Hard | ~40-70% |
| **Overall** | **~60-85%** |

### OpenAI API Baseline (`training/baseline_openai.py`)
Uses `gpt-4o-mini` with zero-shot prompting via the OpenAI API.

```bash
export OPENAI_API_KEY=sk-...
python training/baseline_openai.py
```

| Tier | Expected Accuracy |
|:---|:---|
| Easy (5 tasks) | ~90-100% |
| Medium (5 tasks) | ~70-85% |
| Hard (5 tasks) | ~50-70% |
| **Overall** | **~70-85%** |

> Hard tasks like `hard_damaged` (must call `ask_proof` before refund) and `hard_missing` (requires 4-step chain) genuinely challenge frontier models.

## OpenEnv API

| Endpoint | Method | Description |
|:---|:---|:---|
| `/reset` | POST | Initialize a new scenario session |
| `/step/{session_id}` | POST | Submit an agent action |
| `/state/{session_id}` | GET | Returns current session state (OpenEnv spec) |
| `/health` | GET | Server health check |

## File Structure
```
Openev-CSA/
├── agents/                  # Multi-agent system
│   ├── router.py            # Rule-based intent classifier
│   ├── specialist.py        # Order/Logistics/Finance agents
│   ├── supervisor.py        # Review & escalation agent
│   └── orchestrator.py      # Pipeline coordinator
├── my_env/                  # OpenEnv environment
│   ├── models.py            # Typed Pydantic models
│   ├── client.py            # Python client SDK
│   ├── openenv.yaml         # Environment spec
│   └── server/
│       ├── app.py           # FastAPI server
│       ├── my_environment.py # step()/reset()/state() logic
│       ├── tools.py         # 16 tool implementations
│       └── db.py            # Simulated e-commerce database
├── training/                # RL training pipeline
│   ├── train.py             # GRPO trainer
│   ├── rewards.py           # 6 reward functions
│   ├── rollout.py           # Environment rollout
│   ├── inference.py         # 15-scenario benchmark
│   └── dataset.py           # Training prompts
├── app.py                   # Gradio UI (multi-agent)
├── Dockerfile               # HF Spaces deployment
├── run_app.sh               # Startup script
└── requirements.txt         # Dependencies
```

## Technical Stack
- **Base Model**: `Qwen/Qwen2.5-1.5B`
- **RL Framework**: `trl` (GRPO — Group Relative Policy Optimization)
- **Environment**: FastAPI + Custom OpenEnv spec
- **UI**: Gradio
- **Deployment**: Docker on Hugging Face Spaces
