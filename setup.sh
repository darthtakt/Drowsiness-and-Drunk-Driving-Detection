#!/bin/bash

# Download facial landmark predictor if not exists
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
  echo "Downloading facial landmark predictor model..."
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

# Create a simple alert sound if not exists
if [ ! -f "alert_sound.wav" ]; then
  echo "Creating a simple alert sound..."
  # Use ffmpeg if available
  if command -v ffmpeg &> /dev/null; then
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=2" alert_sound.wav
  else
    echo "FFmpeg not available, will create alert sound using Python"
  fi
fi

# Make sure setup.sh has execution permissions
chmod +x setup.sh