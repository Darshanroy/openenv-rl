# --- Standalone OpenEnv CSA Environment API ---
# Deploy to: Darshankumarr03/openenv-csa-rl
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only environment-related code
COPY my_env ./my_env
COPY openenv.yaml .
COPY pyproject.toml .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# HF Spaces listens on port 7860
EXPOSE 7860

# Start the OpenEnv API server on port 7860
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
