import os
import tempfile
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import our modules
from utils.file_parser import ResumeParser, save_uploaded_file, cleanup_temp_file
from utils.ml_matcher import get_matcher

# Import data collector
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.data_collector import TrainingDataCollector

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['SECRET_KEY'] = os.urandom(24)  # For session management
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Initialize components
resume_parser = ResumeParser()
matcher = get_matcher()  # This will try to load ML models, fall back to similarity
data_collector = TrainingDataCollector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    # Generate session ID for tracking
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Get model info for display
    model_info = matcher.get_model_info()
    
    return render_template('index.html', model_info=model_info)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze resume against job description"""
    try:
        # Check if files and data are present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        if resume_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not job_description.strip():
            return jsonify({'error': 'Job description is required'}), 400
        
        if not allowed_file(resume_file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PDF or DOCX'}), 400
        
        # Save file temporarily
        file_path = save_uploaded_file(resume_file)
        file_extension = resume_file.filename.rsplit('.', 1)[1].lower()
        
        # Extract text from resume
        try:
            resume_text = resume_parser.extract_text(file_path, file_extension)
            sections = resume_parser.extract_sections(resume_text)
        except Exception as e:
            return jsonify({'error': f'Could not extract text from resume: {str(e)}'}), 400
        finally:
            # Clean up temp file
            cleanup_temp_file(file_path)
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from the resume'}), 400
        
        # Analyze using ML matcher
        analysis = matcher.analyze_with_confidence(resume_text, job_description)
        
        # Save to database for future training (if score is reasonable)
        if analysis['score'] > 0:
            data_collector.save_match_result(
                resume_text=resume_text[:1000],  # Save first 1000 chars to save space
                job_description=job_description[:1000],
                ai_score=analysis['score'],
                session_id=session.get('session_id', 'unknown')
            )
        
        # Prepare response
        response = {
            'success': True,
            'score': analysis['score'],
            'keywords': analysis['keywords'][:20],  # Limit to top 20
            'method': analysis['method'],
            'confidence': analysis['confidence'],
            'confidence_level': analysis['confidence_level'],
            'models_used': analysis['models_used'],
            'sections': list(sections.keys()) if sections else [],
            'message': 'Analysis complete'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Collect user feedback on match accuracy"""
    try:
        data = request.json
        session_id = session.get('session_id', data.get('session_id', 'unknown'))
        
        # Save feedback
        data_collector.save_match_result(
            resume_text=data.get('resume_text', '')[:1000],
            job_description=data.get('job_description', '')[:1000],
            ai_score=data.get('ai_score', 0),
            user_score=data.get('user_score'),
            is_accurate=data.get('is_accurate'),
            feedback_text=data.get('feedback', ''),
            session_id=session_id
        )
        
        return jsonify({'success': True, 'message': 'Feedback received. Thank you!'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-success', methods=['POST'])
def save_success():
    """Save a successful match as positive example"""
    try:
        data = request.json
        
        data_collector.save_successful_match(
            resume_text=data.get('resume_text', '')[:1000],
            job_description=data.get('job_description', '')[:1000],
            match_score=data.get('score', 0),
            industry=data.get('industry'),
            role=data.get('role')
        )
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-failure', methods=['POST'])
def save_failure():
    """Save a failed match as negative example"""
    try:
        data = request.json
        
        data_collector.save_failed_match(
            resume_text=data.get('resume_text', '')[:1000],
            job_description=data.get('job_description', '')[:1000],
            match_score=data.get('score', 0),
            reason=data.get('reason', 'user_reported')
        )
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about current model"""
    info = matcher.get_model_info()
    
    # Add training statistics
    try:
        stats = data_collector.get_statistics()
        info['training_data'] = stats
    except:
        info['training_data'] = {'error': 'Could not fetch statistics'}
    
    return jsonify(info)

@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining (admin only - you'd add auth in production)"""
    try:
        from training.train_model import auto_train
        
        # Run auto-training
        trainer, version = auto_train()
        
        # Reload models in matcher
        matcher._load_models()
        
        return jsonify({
            'success': True, 
            'message': f'Models retrained successfully',
            'version': version
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)