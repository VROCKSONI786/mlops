FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libsm6 libxext6 libssl-dev \
        ca-certificates curl gnupg && \
    update-ca-certificates && \
    pip install --no-cache-dir --upgrade pip setuptools wheel certifi requests && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create required dirs
RUN mkdir -p data/raw data/processed models mlruns templates

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "-c", "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=5000, threads=4)"]
