FROM python:3.10.11-slim

# Install system dependencies with retry
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        cmake \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create the directory for the model file
RUN mkdir -p /app/models

# Download shape predictor model with retry mechanism
RUN for i in 1 2 3 4 5; do \
        echo "Attempt $i: Downloading shape predictor model..." && \
        wget -O /app/models/shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
        bzip2 -d /app/models/shape_predictor_68_face_landmarks.dat.bz2 && \
        break || sleep 15; \
    done

# Create a simple alert sound
RUN echo "Creating simple alert sound" && \
    echo "UklGRjIAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAA==" | base64 -d > /app/alert_sound.wav

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (use pre-built dlib wheel)
RUN pip install --no-cache-dir --upgrade pip && \
    sed -i 's/dlib==19.24.2/https:\/\/github.com\/davisking\/dlib\/files\/13779635\/dlib-19.24.2-cp310-cp310-linux_x86_64.whl/g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the model path and sound path
ENV SHAPE_PREDICTOR_PATH=/app/models/shape_predictor_68_face_landmarks.dat
ENV ALERT_SOUND_PATH=/app/alert_sound.wav
ENV PORT=10000

# Run the application
CMD waitress-serve --host=0.0.

