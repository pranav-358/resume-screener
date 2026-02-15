# Resume Screening System using AI/ML

An intelligent system that classifies resumes into job categories and matches them with job descriptions using Machine Learning.

## Features
- 📄 **Resume Classification**: Automatically categorizes resumes (Java Developer, Data Scientist, etc.)
- 🎯 **Job Matching**: Calculates similarity score between resume and job description
- 📊 **Confidence Scoring**: Shows prediction confidence percentage
- 🔍 **PDF Support**: Extracts and analyzes text from PDF resumes
- 🎨 **Modern UI**: Clean interface with Tailwind CSS

## Tech Stack
- **Backend**: Flask, Gunicorn
- **ML/AI**: scikit-learn, NLTK, pandas, numpy
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **PDF Processing**: PyPDF2, pdfplumber
- **Deployment**: Render

## Dataset Format
Create `UpdatedResumeDataSet.csv` in your root folder:
```csv
Category,Resume
Java Developer,"Experienced Java developer with Spring Boot, Hibernate, Microservices"
Data Scientist,"PhD in Machine Learning, Python, TensorFlow, NLP"
Python Developer,"Full stack Python developer with Django, Flask, FastAPI"
Frontend Developer,"React expert with TypeScript, Redux, modern CSS"
DevOps Engineer,"AWS certified, Docker, Kubernetes, Jenkins, Terraform"

## Quick Start

### Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/resume-screener.git
cd resume-screener

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run app
python app.py```

##Project Structure
text
resume-screener/
├── app.py                 # Flask backend application
├── train_model.py         # ML training pipeline
├── requirements.txt       # Project dependencies
├── render.yaml            # Render deployment config
├── models/                # Trained ML models
│   ├── classifier.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── templates/             # Frontend files
│   └── index.html        # Main UI
└── uploads/               # Temporary PDF storage

Performance Metrics
Accuracy: 98% on test data

Categories: 25+ job roles supported

Response Time: < 2 seconds per prediction

File Support: PDF up to 16MB

Confidence Scoring: Yes, with percentage

##Live Demo
-https://resume-screener.onrender.com

-Deployment on Render
-Push code to GitHub

-Create new Web Service on Render

##Use these settings:

Build Command: pip install -r requirements.txt && python train_model.py

Start Command: gunicorn app:app

Environment Variable: PYTHON_VERSION = 3.11.5

API Endpoints
Endpoint	Method	Description
/	GET	Web interface
/predict	POST	Upload & classify resume
/match	POST	Calculate job match
/health	GET	Health check
API Usage Examples
bash
# Predict resume category
curl -X POST -F "resume=@resume.pdf" https://resume-screener.onrender.com/predict

# Match with job description
curl -X POST -H "Content-Type: application/json" \
  -d '{"job_description":"Looking for Python developer with 5 years experience..."}' \
  https://resume-screener.onrender.com/match
Environment Variables
Create a .env file for local development:

text
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
License
MIT

Author
Your Name - Pranav Tayade

Project Link
https://github.com/yourusername/resume-screener
