# train_model.py
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors  # Changed import
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression  # Alternative classifier

# Download NLTK data (run once)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class ResumeTrainer:
    def __init__(self, csv_path='UpdatedResumeDataSet.csv'):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load the CSV dataset"""
        print("📂 Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df)} resumes")
        print(f"📊 Categories: {self.df['Category'].nunique()}")
        return self.df
    
    def clean_text(self, text):
        """Clean resume text by removing URLs, special chars, etc."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
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
        
        # Tokenize and lemmatize (simplified for speed)
        words = text.split()[:1000]  # Limit to first 1000 words for speed
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_data(self):
        """Clean all resumes in the dataset"""
        print("🧹 Cleaning resume text...")
        # Apply cleaning with progress indicator
        from tqdm import tqdm
        tqdm.pandas()
        self.df['Cleaned_Resume'] = self.df['Resume'].progress_apply(self.clean_text)
        
        # Remove empty resumes
        self.df = self.df[self.df['Cleaned_Resume'].str.len() > 0].reset_index(drop=True)
        
        # Encode categories
        self.df['Category_Code'] = self.label_encoder.fit_transform(self.df['Category'])
        
        # Create category mapping
        self.category_mapping = dict(zip(
            self.df['Category_Code'], 
            self.df['Category']
        ))
        
        print(f"✅ Preprocessing complete. {len(self.df)} valid resumes")
        return self.df
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the TF-IDF + Classifier model"""
        print("🤖 Training model...")
        
        # Split data
        X = self.df['Cleaned_Resume']
        y = self.df['Category_Code']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Option 1: Using Logistic Regression (more robust with sparse matrices)
        print("Using Logistic Regression classifier...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                max_df=0.8,
                min_df=2,
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='ovr',
                n_jobs=-1,
                random_state=random_state
            ))
        ])
        
        # Train
        print("Fitting pipeline...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with accuracy: {accuracy:.2%}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return self.pipeline
    
    def save_models(self, model_path='models/classifier.pkl'):
        """Save the trained pipeline and encoders"""
        print("💾 Saving models...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the entire pipeline
        joblib.dump(self.pipeline, model_path)
        
        # Save label encoder separately
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        # Save category mapping
        joblib.dump(self.category_mapping, 'models/category_mapping.pkl')
        
        # Also save just the vectorizer for potential reuse
        joblib.dump(self.pipeline.named_steps['tfidf'], 'models/vectorizer.pkl')
        
        print(f"✅ Models saved to models/ directory")
        print(f"   - classifier.pkl (full pipeline)")
        print(f"   - vectorizer.pkl (TF-IDF vectorizer)")
        print(f"   - label_encoder.pkl")
        print(f"   - category_mapping.pkl")
    
    def test_prediction(self, sample_text=None):
        """Test a sample prediction"""
        if sample_text is None:
            # Take a random sample from the dataset
            sample_idx = np.random.randint(0, len(self.df))
            sample_text = self.df.iloc[sample_idx]['Resume']
            true_category = self.df.iloc[sample_idx]['Category']
            
            print(f"\n🔍 Testing with sample resume (True category: {true_category})")
        
        # Clean and predict
        cleaned = self.clean_text(sample_text)
        pred_code = self.pipeline.predict([cleaned])[0]
        pred_category = self.label_encoder.inverse_transform([pred_code])[0]
        
        # Get prediction probabilities
        if hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            probs = self.pipeline.predict_proba([cleaned])[0]
            confidence = max(probs) * 100
            print(f"✅ Predicted: {pred_category} (Confidence: {confidence:.2f}%)")
        else:
            print(f"✅ Predicted: {pred_category}")
        
        return pred_category
    
    def run(self):
        """Execute the complete training pipeline"""
        print("="*50)
        print("🚀 RESUME SCREENING - TRAINING PIPELINE")
        print("="*50)
        
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.save_models()
        self.test_prediction()
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETE! Ready for Flask app.")
        print("="*50)
        
        # Show sample categories
        print("\n📋 Sample Categories (first 10):")
        for i, (code, category) in enumerate(list(self.category_mapping.items())[:10]):
            print(f"  {code}: {category}")

if __name__ == "__main__":
    trainer = ResumeTrainer()
    trainer.run()