#!/usr/bin/env python
"""
Complete Training Pipeline
Run this to train and evaluate models periodically
"""
import os
import sys
import argparse
import schedule
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.data_collector import TrainingDataCollector
from training.train_model import ResumeMatcherTrainer

class TrainingPipeline:
    """
    Automated training pipeline for resume matcher
    """
    
    def __init__(self, min_samples_for_training=50):
        self.collector = TrainingDataCollector()
        self.trainer = ResumeMatcherTrainer()
        self.min_samples = min_samples_for_training
    
    def check_and_train(self):
        """
        Check if enough data is available and train if needed
        """
        logger.info("🔍 Checking training data status...")
        
        # Get statistics
        stats = self.collector.get_statistics()
        
        total_samples = (
            stats['successful_matches'] + 
            stats['failed_matches'] + 
            stats['with_feedback']
        )
        
        logger.info(f"   Total training samples: {total_samples}")
        logger.info(f"   Required minimum: {self.min_samples}")
        
        if total_samples >= self.min_samples:
            logger.info("✅ Enough samples available, starting training...")
            return self.run_training()
        else:
            logger.warning(f"⚠️ Not enough samples. Need {self.min_samples - total_samples} more.")
            return False
    
    def run_training(self, test_split=0.2):
        """
        Run the full training pipeline
        """
        try:
            logger.info("=" * 50)
            logger.info("🚀 Starting Training Pipeline")
            logger.info("=" * 50)
            
            # Export data to CSV
            logger.info("📤 Exporting training data...")
            data_path = f'training/data/training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df = self.collector.export_to_csv(data_path)
            logger.info(f"   Exported {len(df)} samples to {data_path}")
            
            # Split into train/test
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_split, random_state=42)
            
            train_path = data_path.replace('.csv', '_train.csv')
            test_path = data_path.replace('.csv', '_test.csv')
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"   Training set: {len(train_df)} samples")
            logger.info(f"   Test set: {len(test_df)} samples")
            
            # Train models
            logger.info("🎯 Training models...")
            models, version = self.trainer.train_models(train_path)
            
            # Evaluate on test set
            logger.info("📊 Evaluating on test set...")
            results = self.trainer.evaluate_on_test_set(test_path)
            
            # Log results
            logger.info("✅ Training Results:")
            for key, value in results.items():
                logger.info(f"   {key}: {value}")
            
            # Save training report
            report_path = f'training/reports/training_report_{version}.txt'
            os.makedirs('training/reports', exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(f"Training Report - {datetime.now()}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Version: {version}\n")
                f.write(f"Training samples: {len(train_df)}\n")
                f.write(f"Test samples: {len(test_df)}\n")
                f.write("\nResults:\n")
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"📝 Report saved to {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def schedule_training(self, interval_hours=24):
        """
        Schedule periodic training
        """
        logger.info(f"⏰ Scheduling training every {interval_hours} hours")
        
        # Run once immediately
        self.check_and_train()
        
        # Schedule periodic runs
        schedule.every(interval_hours).hours.do(self.check_and_train)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(description='Training Pipeline for Resume Matcher')
    parser.add_argument('--run-now', action='store_true', help='Run training immediately')
    parser.add_argument('--schedule', type=int, metavar='HOURS', 
                       help='Schedule training every N hours')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum samples required for training')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(min_samples_for_training=args.min_samples)
    
    if args.run_now:
        pipeline.run_training()
    
    if args.schedule:
        pipeline.schedule_training(args.schedule)
    
    if not args.run_now and not args.schedule:
        # Just check status
        stats = pipeline.collector.get_statistics()
        print("\n📊 Current Training Data Status:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        total = stats['successful_matches'] + stats['failed_matches'] + stats['with_feedback']
        print(f"\n   Total training samples: {total}")
        print(f"   Required: {args.min_samples}")
        
        if total >= args.min_samples:
            print("\n✅ Ready for training! Run with --run-now")
        else:
            print(f"\n⚠️ Need {args.min_samples - total} more samples for training")

if __name__ == "__main__":
    main()