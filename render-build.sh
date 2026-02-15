#!/bin/bash
# render-build.sh - FIXED for Python 3.11
python -m pip install --upgrade pip setuptools wheel
echo "🚀 Starting build for Python 3.11..."

# CRITICAL: Install setuptools FIRST (this fixes pkg_resources error)
python -m pip install --upgrade pip setuptools wheel

# Now install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Train model
python train_model.py

echo "✅ Build complete!"