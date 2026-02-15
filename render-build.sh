#!/bin/bash
# render-build.sh

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python nltk_setup.py

# Train the model
python train_model.py

echo "Build complete!"