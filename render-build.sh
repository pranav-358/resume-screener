#!/bin/bash
echo "🚀 Starting build..."

# Show Python version for debugging
python --version

# Force install setuptools first
python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel==0.41.2

# Now install everything else
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Train model
python train_model.py

echo "✅ Build complete!"