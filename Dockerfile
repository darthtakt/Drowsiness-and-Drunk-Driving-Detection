FROM python:3.10.11-slim

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy setup script and run it
COPY setup.sh .
RUN chmod +x setup.sh && ./setup.sh

# Copy application code
COPY . .

# Set environment variables
ENV PORT=10000

# Run the application
CMD waitress-serve --host=0.0.0.0 --port=$PORT app:app
