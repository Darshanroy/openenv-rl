FROM python:3.10-slim

WORKDIR /app

# Copy generic requirement definitions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . /app/my_env/server/
# Copy models
COPY ../models.py /app/my_env/

ENV PYTHONPATH=/app

CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
