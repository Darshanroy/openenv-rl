---
title: OpenEnv CSA Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# OpenEnv CSA — Customer Support Environment

Standalone OpenEnv-compliant environment API for the Customer Support Agent.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/reset` | POST | Reset environment (OpenEnv spec) |
| `/step` | POST | Execute one action (OpenEnv spec) |
| `/session/reset` | POST | Start a session-based episode |
| `/session/step/{id}` | POST | Step a session-based episode |

## Usage

Point your `inference.py` at this Space:

```python
ENV_URL = "https://darshankumarr03-openenv-csa-rl.hf.space"
```
