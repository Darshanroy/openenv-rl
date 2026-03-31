#!/bin/bash

# Ensure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:.

# Start the environment server in the background
echo "Starting OpenEnv API Server..."
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000 &

# Start the main Gradio application in the foreground
echo "Starting Gradio UI..."
python app.py
