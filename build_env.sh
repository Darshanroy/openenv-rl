#!/bin/bash

# --- Build the Standalone OpenEnv CSA Environment ---
IMAGE_NAME="openenv-csa-env"

echo "🛠️ Building Standalone Environment Image..."
docker build -f Dockerfile.env -t ${IMAGE_NAME}:latest .

echo "✅ Image '${IMAGE_NAME}:latest' is ready for inference.py"
echo "--------------------------------------------------------"
echo "To run the evaluation, simply execute:"
echo "python inference.py"
echo "--------------------------------------------------------"
