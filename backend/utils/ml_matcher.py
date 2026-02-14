"""
ML-Enhanced Matcher Module
Uses trained models for resume matching with fallback to similarity
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class MLResumeMatcher:
    """
    Enhanced resume matcher that uses trained ML models
    Falls back to cosine similarity if models aren't available
    """
    
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        self.model_loaded = False
        self.use_ml = False
        
        # Try to load ML models
        self._load_models()
        
        # Always have fallback vectorizer ready
        self.fallback_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
    
    def _load_models(self):
        """Load trained ML models if available"""
        try:
            # Try to load latest models
            self.classifier = joblib.load(f"{self.model_dir}/classifier_latest.pkl")
            self.regressor = joblib.load(f"{self.model_dir}/regressor_latest.pkl")
            self.vectorizer = joblib.load(f"{self.model_dir}/vectorizer_latest.pkl")
            self.model_loaded = True
            self.use_ml = True
            print("✅ ML models loaded successfully")
        except FileNotFoundError:
            print("⚠️ No trained models found, using similarity-based matching")
            self.model_loaded = False
            self.use_ml = False
        except Exception as e:
            print(f"⚠️ Error loading models: {e}, using similarity-based matching")
            self.model_loaded = False
            self.use_ml = False
    
    def predict_match(self, resume_text, job_description):
        """
        Predict match score using best available method
        """
        if self.use_ml and self.model_loaded:
            return self._ml_predict(resume_text, job_description)
        else:
            return self._similarity_fallback(resume_text, job_description)
    
    def _ml_predict(self, resume_text, job_description):
        """
        Use ML models for prediction
        """
        try:
            # Combine texts
            combined = resume_text + " [SEP] " + job_description
            
            # Vectorize
            X = self.vectorizer.transform([combined])
            
            # Try regression first (gives score 0-100)
            if hasattr(self, 'regressor'):
                score = float(self.regressor.predict(X)[0])
                
                # Ensure score is within bounds
                score = max(0, min(100, score))
                
                # Get confidence from classifier if available
                confidence = None
                if hasattr(self, 'classifier'):
                    proba = self.classifier.predict_proba(X)[0]
                    confidence = float(max(proba))
                
                return {
                    'score': round(score, 2),
                    'method': 'ml_regression',
                    'confidence': confidence,
                    'models_used': ['regressor']
                }
            
            # Fallback to classifier (binary)
            elif hasattr(self, 'classifier'):
                prediction = self.classifier.predict(X)[0]
                proba = self.classifier.predict_proba(X)[0]
                
                # Convert binary to score (0 or 100)
                score = 100 if prediction == 1 else 0
                confidence = float(max(proba))
                
                return {
                    'score': score,
                    'method': 'ml_classification',
                    'confidence': confidence,
                    'models_used': ['classifier']
                }
            
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}, falling back to similarity")
            return self._similarity_fallback(resume_text, job_description)
    
    def _similarity_fallback(self, resume_text, job_description):
        """
        Fallback to cosine similarity
        """
        try:
            # Preprocess texts
            resume_processed = self._preprocess_text(resume_text)
            job_processed = self._preprocess_text(job_description)
            
            # Create TF-IDF vectors
            documents = [resume_processed, job_processed]
            tfidf_matrix = self.fallback_vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            # Convert to percentage
            score = float(cosine_sim[0][0]) * 100
            
            return {
                'score': round(score, 2),
                'method': 'cosine_similarity',
                'confidence': None,
                'models_used': ['tfidf']
            }
            
        except Exception as e:
            print(f"⚠️ Similarity calculation failed: {e}")
            return {
                'score': 0,
                'method': 'failed',
                'confidence': None,
                'models_used': [],
                'error': str(e)
            }
    
    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_matched_keywords(self, resume_text, job_description, top_n=20):
        """
        Extract keywords that match between resume and job description
        """
        # Preprocess texts
        resume_words = set(self._extract_important_words(resume_text))
        job_words = set(self._extract_important_words(job_description))
        
        # Find intersection
        matched = resume_words.intersection(job_words)
        
        # Sort by importance (frequency in job description)
        job_word_freq = self._get_word_frequencies(job_description, matched)
        sorted_keywords = sorted(matched, 
                               key=lambda x: job_word_freq.get(x, 0), 
                               reverse=True)
        
        return sorted_keywords[:top_n]
    
    def _extract_important_words(self, text, max_words=100):
        """Extract important words using simple frequency"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Simple stop words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 
                     'from', 'your', 'will', 'are', 'can', 'all', 'not',
                     'but', 'our', 'their', 'they', 'been', 'has', 'had'}
        
        # Filter and count
        important_words = [w for w in words if w not in stop_words]
        
        # Get most common
        from collections import Counter
        word_counts = Counter(important_words)
        
        return [word for word, count in word_counts.most_common(max_words)]
    
    def _get_word_frequencies(self, text, words):
        """Get frequency of specific words in text"""
        text_lower = text.lower()
        frequencies = {}
        
        for word in words:
            # Count occurrences
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            frequencies[word] = count
        
        return frequencies
    
    def analyze_with_confidence(self, resume_text, job_description):
        """
        Perform analysis with confidence scores
        """
        # Get prediction
        result = self.predict_match(resume_text, job_description)
        
        # Get keywords
        keywords = self.get_matched_keywords(resume_text, job_description)
        
        # Calculate confidence level
        if result['confidence']:
            confidence_level = 'high' if result['confidence'] > 0.8 else 'medium' if result['confidence'] > 0.6 else 'low'
        else:
            confidence_level = 'unknown'
        
        return {
            'score': result['score'],
            'method': result['method'],
            'confidence': result['confidence'],
            'confidence_level': confidence_level,
            'keywords': keywords,
            'models_used': result['models_used']
        }
    
    def get_model_info(self):
        """Get information about loaded models"""
        if self.use_ml and self.model_loaded:
            return {
                'status': 'active',
                'type': 'ml_models',
                'classifier': hasattr(self, 'classifier'),
                'regressor': hasattr(self, 'regressor'),
                'vectorizer': hasattr(self, 'vectorizer')
            }
        else:
            return {
                'status': 'fallback',
                'type': 'cosine_similarity',
                'message': 'Using similarity-based matching (no ML models)'
            }


# Factory function
def get_matcher(model_dir='models/'):
    """
    Get configured matcher instance
    """
    return MLResumeMatcher(model_dir)


# Quick test
if __name__ == "__main__":
    # Test the matcher
    matcher = MLResumeMatcher()
    
    # Sample texts
    resume = "Python developer with 5 years Django experience"
    job = "Looking for Python developer with Django skills"
    
    # Analyze
    result = matcher.analyze_with_confidence(resume, job)
    
    print("\n📊 Analysis Result:")
    print(f"   Score: {result['score']}%")
    print(f"   Method: {result['method']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Keywords: {result['keywords'][:5]}")
    print(f"   Models: {result['models_used']}")
    
    print("\n🤖 Model Info:")
    print(matcher.get_model_info())