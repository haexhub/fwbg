FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY *.py .
COPY mapping.yaml .
# config.yaml wird über Volume gemountet (enthält Secrets)

# Create directories for runtime data
RUN mkdir -p data stats_export

# Run the bot
CMD ["python", "ig_bot.py"]
