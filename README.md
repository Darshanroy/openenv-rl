---
title: OpenEnv CSA - RL Customer Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv Customer Support Agent (CSA)

This Space hosts the OpenEnv Customer Support Agent (CSA), a Reinforcement Learning agent trained to handle complex e-commerce scenarios.

## Features
- **15 Benchmark Scenarios**: Test the agent on tasks ranging from simple status checks to hard escalations.
- **Simulated Environment**: An internal FastAPI server simulates the customer database and tool executions.
- **Gradio UI**: Interactive chat interface for real-time model interaction.

## Technical Stack
- **Base Model**: `Qwen/Qwen2.5-1.5B`
- **Framework**: `trl` (GRPO training), `transformers`, `FastAPI`, `Gradio`
- **Deployment**: Dockerized multi-process setup.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
