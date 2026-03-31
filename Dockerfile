# Use high-performance Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install system dependencies (git is required for TRL installation)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure startup scripts are executable
RUN chmod +x run_app.sh

# Hugging Face Spaces run on port 7860 by default
EXPOSE 7860

# Start the multi-process application
CMD ["./run_app.sh"]
