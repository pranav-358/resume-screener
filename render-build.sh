#!/bin/bash
# render-build.sh - THE SIMPLE, GUARANTEED FIX

echo "🚀 Starting build for Python 3.11..."

# STEP 1: Upgrade pip and install essential build tools FIRST
# This single line fixes the 'pkg_resources' error
python -m pip install --upgrade pip setuptools wheel

# STEP 2: Now install your project dependencies
pip install -r requirements.txt

# STEP 3: Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# STEP 4: Train your model
python train_model.py

echo "✅ Build complete!"