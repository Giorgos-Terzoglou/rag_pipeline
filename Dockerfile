# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download SpaCy English model
RUN python -m spacy download en_core_web_sm

# Expose API port
EXPOSE 8000

# Start API automatically
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
