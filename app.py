# app.py
import os
import joblib
import PyPDF2
import numpy as np
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdfplumber
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models at startup
print("🔄 Loading trained models...")
try:
    # Load the full pipeline
    pipeline = joblib.load('models/classifier.pkl')
    
    # Load label encoder and mapping
    label_encoder = joblib.load('models/label_encoder.pkl')
    category_mapping = joblib.load('models/category_mapping.pkl')
    
    # Extract components
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    
    print("✅ Models loaded successfully!")
    print(f"📊 Model type: {type(classifier).__name__}")
    print(f"📊 Categories: {len(category_mapping)}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("Please run train_model.py first!")
    pipeline = None
    vectorizer = None
    classifier = None
    label_encoder = None
    category_mapping = {}
except Exception as e:
    print(f"❌ Unexpected error loading models: {e}")
    pipeline = None
    vectorizer = None
    classifier = None
    label_encoder = None
    category_mapping = {}

# Initialize text cleaner
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize (limit for speed)
        words = text.split()[:500]  # Limit to first 500 words
        words = [lemmatizer.lemmatize(word) for word in words 
                if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        # Try pdfplumber first (better for tables/complex layouts)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Fallback to PyPDF2 if pdfplumber fails or returns little text
        if len(text.strip()) < 100:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        return ""

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle resume upload and prediction"""
    try:
        # Check if file was uploaded
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        raw_text = extract_text_from_pdf(filepath)
        
        if not raw_text or len(raw_text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from PDF'}), 400
        
        # Clean the text
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text:
            return jsonify({'error': 'Text cleaning failed'}), 400
        
        # Make prediction
        if pipeline is None:
            return jsonify({'error': 'Model not loaded. Please train first.'}), 500
        
        # Predict category
        category_code = pipeline.predict([cleaned_text])[0]
        
        # Convert code to category name
        if label_encoder is not None:
            category_name = label_encoder.inverse_transform([category_code])[0]
        else:
            category_name = category_mapping.get(int(category_code), "Unknown")
        
        # Get prediction probabilities/confidence
        confidence = 0.0
        if hasattr(classifier, 'predict_proba'):
            try:
                probabilities = pipeline.predict_proba([cleaned_text])[0]
                confidence = float(max(probabilities) * 100)
            except:
                # Fallback confidence calculation
                confidence = 85.0  # Default confidence
        else:
            # For models without predict_proba, use decision function
            try:
                if hasattr(classifier, 'decision_function'):
                    decision_scores = classifier.decision_function(
                        vectorizer.transform([cleaned_text])
                    )
                    # Convert decision scores to pseudo-probabilities
                    if len(decision_scores.shape) > 1:
                        confidence = float(
                            (np.max(decision_scores) - np.min(decision_scores)) * 50
                        )
                    else:
                        confidence = float(abs(decision_scores[0]) * 10)
                else:
                    confidence = 85.0
            except:
                confidence = 85.0
        
        # Clean up - remove uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Store resume text in session for job matching
        session['last_resume'] = cleaned_text
        
        return jsonify({
            'success': True,
            'category': category_name,
            'confidence': round(min(confidence, 99.99), 2),
            'message': f'Resume classified as {category_name}'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['POST'])
def match_job():
    """Calculate match score between resume and job description"""
    try:
        data = request.get_json()
        
        if not data or 'job_description' not in data:
            return jsonify({'error': 'No job description provided'}), 400
        
        job_desc = data['job_description']
        
        if not job_desc or len(job_desc.strip()) < 20:
            return jsonify({'error': 'Job description too short'}), 400
        
        # Get the last processed resume from session
        resume_text = session.get('last_resume')
        
        if not resume_text:
            return jsonify({'error': 'No resume found. Please upload a resume first.'}), 400
        
        # Clean job description
        cleaned_job = clean_text(job_desc)
        
        if not cleaned_job:
            return jsonify({'error': 'Invalid job description'}), 400
        
        # Calculate similarity
        if vectorizer is None:
            return jsonify({'error': 'Vectorizer not loaded'}), 500
        
        # Transform both texts
        try:
            texts = [resume_text, cleaned_job]
            tfidf_matrix = vectorizer.transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Handle potential NaN
            if np.isnan(similarity):
                similarity = 0.0
                
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return jsonify({'error': 'Error calculating similarity'}), 500
        
        # Convert to percentage
        match_percentage = round(float(similarity) * 100, 2)
        
        # Generate insights based on match percentage
        insights = []
        if match_percentage >= 80:
            insights.append("🌟 Excellent match! The candidate's skills align perfectly with the job requirements.")
            insights.append("📊 Strong keyword overlap indicates good fit for this role.")
        elif match_percentage >= 60:
            insights.append("👍 Good match. Core requirements are met, but some skills may need development.")
            insights.append("📈 Consider checking for specific technical skills in the resume.")
        elif match_percentage >= 40:
            insights.append("⚠️ Moderate match. The candidate has some relevant experience but may need training.")
            insights.append("🎯 Review the resume for transferable skills that might not be captured in the similarity score.")
        else:
            insights.append("❌ Low match. The candidate may not be suitable for this specific role.")
            insights.append("💡 Consider other positions that better align with their experience.")
        
        return jsonify({
            'success': True,
            'match_score': match_percentage,
            'insights': insights,
            'message': f'Match score: {match_percentage}%'
        })
        
    except Exception as e:
        logger.error(f"Match error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline is not None,
        'categories': len(category_mapping) if category_mapping else 0,
        'model_type': type(classifier).__name__ if classifier else None
    })

if __name__ == '__main__':
   port = int(os.environ.get('PORT', 5000))
   app.run(host='0.0.0.0', port=port, debug=False)