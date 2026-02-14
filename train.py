#!/usr/bin/env python
"""
Training Script Runner
Run this script to train the resume matching models
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.data_collector import TrainingDataCollector, SampleDataCreator
from training.train_model import ResumeMatcherTrainer, auto_train

def main():
    parser = argparse.ArgumentParser(description='Train Resume Matching Models')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample dataset for initial training')
    parser.add_argument('--export-data', action='store_true',
                       help='Export collected data to CSV')
    parser.add_argument('--train', action='store_true',
                       help='Train models using collected data')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-train (create sample if needed, then train)')
    parser.add_argument('--stats', action='store_true',
                       help='Show training data statistics')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Resume Matcher Training System")
    print("=" * 60)
    
    # Initialize components
    collector = TrainingDataCollector()
    
    if args.stats:
        # Show statistics
        stats = collector.get_statistics()
        print("\n📊 Training Data Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return
    
    if args.create_sample:
        # Create sample dataset
        print("\n📝 Creating sample dataset...")
        SampleDataCreator.create_sample_dataset()
    
    if args.export_data:
        # Export data
        print("\n📤 Exporting training data...")
        df = collector.export_to_csv()
        print(f"   Exported {len(df)} samples")
    
    if args.train:
        # Train models
        print("\n🎯 Training models...")
        trainer = ResumeMatcherTrainer()
        models, version = trainer.train_models()
        print(f"   Models saved with version: {version}")
    
    if args.auto:
        # Auto-train
        print("\n🤖 Running auto-training...")
        trainer, version = auto_train()
    
    if not any([args.create_sample, args.export_data, args.train, args.auto, args.stats]):
        parser.print_help()

if __name__ == "__main__":
    main()