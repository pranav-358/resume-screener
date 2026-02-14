"""
Data Collector Module
Collects and manages training data from user interactions
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

class TrainingDataCollector:
    """
    Collects training data from user interactions for model retraining
    """
    
    def __init__(self, db_path='training/data/training_data.db'):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for storing match results with feedback
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_text TEXT,
                job_description TEXT,
                ai_score FLOAT,
                user_score INTEGER,
                is_accurate BOOLEAN,
                feedback_text TEXT,
                created_at TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # Table for storing successful matches (positive examples)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS successful_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_text TEXT,
                job_description TEXT,
                match_score INTEGER,
                industry TEXT,
                role TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Table for storing failed matches (negative examples)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_text TEXT,
                job_description TEXT,
                match_score INTEGER,
                reason TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized at {self.db_path}")
    
    def save_match_result(self, resume_text, job_description, ai_score, 
                          user_score=None, is_accurate=None, 
                          feedback_text=None, session_id=None):
        """
        Save a match result with optional user feedback
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO matches 
            (resume_text, job_description, ai_score, user_score, 
             is_accurate, feedback_text, created_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resume_text[:5000],  # Limit text length
            job_description[:5000],
            ai_score,
            user_score,
            is_accurate,
            feedback_text,
            datetime.now(),
            session_id
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Match result saved to database")
    
    def save_successful_match(self, resume_text, job_description, 
                               match_score, industry=None, role=None):
        """
        Save a successful match as a positive training example
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO successful_matches 
            (resume_text, job_description, match_score, industry, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            resume_text[:5000],
            job_description[:5000],
            match_score,
            industry,
            role,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Successful match saved as positive example")
    
    def save_failed_match(self, resume_text, job_description, 
                           match_score, reason=None):
        """
        Save a failed match as a negative training example
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO failed_matches 
            (resume_text, job_description, match_score, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            resume_text[:5000],
            job_description[:5000],
            match_score,
            reason,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Failed match saved as negative example")
    
    def export_to_csv(self, output_path='training/data/training_data.csv'):
        """
        Export collected data to CSV for model training
        """
        conn = sqlite3.connect(self.db_path)
        
        # Combine positive and negative examples
        query = '''
            SELECT 
                resume_text,
                job_description,
                match_score as target_score,
                'success' as label,
                1 as is_good_match
            FROM successful_matches
            
            UNION ALL
            
            SELECT 
                resume_text,
                job_description,
                match_score as target_score,
                'failure' as label,
                0 as is_good_match
            FROM failed_matches
            
            UNION ALL
            
            SELECT 
                resume_text,
                job_description,
                COALESCE(user_score, ai_score) as target_score,
                CASE 
                    WHEN is_accurate = 1 THEN 'accurate'
                    WHEN is_accurate = 0 THEN 'inaccurate'
                    ELSE 'unknown'
                END as label,
                is_accurate as is_good_match
            FROM matches
            WHERE user_score IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_path, index=False)
        
        conn.close()
        
        print(f"✅ Training data exported to {output_path}")
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    def get_statistics(self):
        """Get statistics about collected data"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Count matches
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        stats['total_matches'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM successful_matches")
        stats['successful_matches'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM failed_matches")
        stats['failed_matches'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM matches WHERE user_score IS NOT NULL")
        stats['with_feedback'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(ai_score) FROM matches")
        stats['avg_ai_score'] = round(cursor.fetchone()[0] or 0, 2)
        
        conn.close()
        
        return stats


# Sample dataset creator for initial training
class SampleDataCreator:
    """
    Creates sample training data for initial model training
    """
    
    @staticmethod
    def create_sample_dataset(output_path='training/data/sample_training_data.csv'):
        """
        Create a sample dataset with synthetic resumes and job descriptions
        """
        sample_data = [
            # Format: resume_text, job_description, match_score (0-100), is_good_match (0/1)
            
            # Python Developer matches
            (
                "Experienced Python developer with 5 years in Django and Flask. "
                "Built REST APIs and microservices. Proficient in SQL and MongoDB.",
                
                "Senior Python Developer needed with Django experience. "
                "Must know REST APIs and databases.",
                85, 1
            ),
            
            # Java Developer matches
            (
                "Java developer with Spring Boot experience. "
                "Worked on microservices and cloud deployments.",
                
                "Looking for Java developer with Spring framework knowledge. "
                "Experience with microservices preferred.",
                82, 1
            ),
            
            # Mismatch examples
            (
                "Marketing graduate with social media experience. "
                "Created content for Instagram and Facebook.",
                
                "Senior Software Engineer with 10 years experience in C++. "
                "Must have embedded systems background.",
                12, 0
            ),
            
            # Frontend matches
            (
                "Frontend developer skilled in React, Vue, and Angular. "
                "Experienced with responsive design and Tailwind CSS.",
                
                "Frontend Developer needed for React projects. "
                "Must know modern CSS and JavaScript.",
                88, 1
            ),
            
            # Data Science matches
            (
                "Data Scientist with Python, pandas, and scikit-learn experience. "
                "Built machine learning models for prediction.",
                
                "Data Science role requiring Python, ML, and statistical analysis. "
                "Experience with scikit-learn required.",
                90, 1
            ),
            
            # Partial matches
            (
                "Recent graduate with Python knowledge from coursework. "
                "Some experience with Django from academic projects.",
                
                "Mid-level Python Developer with 3+ years commercial experience. "
                "Must have deployed production applications.",
                45, 0
            ),
            
            # Add more samples as needed
        ]
        
        df = pd.DataFrame(sample_data, 
                         columns=['resume_text', 'job_description', 
                                 'match_score', 'is_good_match'])
        
        df.to_csv(output_path, index=False)
        print(f"✅ Sample dataset created at {output_path}")
        print(f"   Samples: {len(df)}")
        
        return df


# Quick test
if __name__ == "__main__":
    # Test the collector
    collector = TrainingDataCollector()
    
    # Create sample data
    SampleDataCreator.create_sample_dataset()
    
    # Test saving a match
    collector.save_match_result(
        resume_text="Sample resume text...",
        job_description="Sample job description...",
        ai_score=75.5,
        session_id="test_session"
    )
    
    # Get statistics
    print("\n📊 Database Statistics:")
    print(collector.get_statistics())
    
    # Export to CSV
    collector.export_to_csv()