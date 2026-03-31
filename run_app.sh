#!/bin/bash

# Ensure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:.

# Start the environment server (OpenEnv API)
echo "🚀 Starting OpenEnv API Server on port 8000..."
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for API Server..."
until curl -s http://localhost:8000/health > /dev/null; do
  sleep 1
done
echo "✅ API Server is up!"

# Start the Streamlit Dashboard (Internal Port 8501)
echo "🚀 Starting Streamlit Dashboard on port 8501..."
streamlit run app.py &
STREAMLIT_PID=$!

# Start NGINX Reverse Proxy (Public Port 7860)
echo "🚀 Starting NGINX Reverse Proxy..."
# Disable daemonizing so NGINX runs in the foreground, keeping the container alive
nginx -g "daemon off;"

# Cleanup on exit
trap "kill $SERVER_PID; kill $STREAMLIT_PID" EXIT
