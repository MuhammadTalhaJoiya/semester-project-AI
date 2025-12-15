"""
Main entry point for Heart Disease Predictor.
Command-line interface for training, prediction, and exploration.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_dataset
from preprocessor import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
from explore_data import DataExplorer
from predict import HeartDiseasePredictor, predict_from_dict


def train_command(args):
    """Train models and save the best one."""
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTOR - TRAINING MODE")
    print("="*60)
    
    try:
        # Load dataset
        print("\n[1/5] Loading dataset...")
        df = load_dataset(args.dataset)
        
        # Preprocess data
        print("\n[2/5] Preprocessing data...")
        preprocessor = DataPreprocessor(test_size=args.test_size, random_state=args.random_state)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df, args.target)
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        
        # Train models
        print("\n[3/5] Training models...")
        trainer = ModelTrainer(random_state=args.random_state)
        trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test, cv=args.cv)
        
        # Evaluate models
        print("\n[4/5] Evaluating models...")
        evaluator = ModelEvaluator()
        comparison = evaluator.evaluate_all_models(trained_models, X_test, y_test)
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        print(comparison.to_string(index=False))
        
        # Generate all plots and reports
        print("\n[5/5] Generating evaluation plots and reports...")
        evaluator.generate_all_plots(trained_models, X_test, y_test)
        
        # Save models
        trainer.save_best_model()
        if args.save_all:
            trainer.save_all_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best accuracy: {trainer.model_scores[trainer.best_model_name]['test_accuracy']:.4f}")
        print(f"\nModels and reports saved to: models/")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def predict_command(args):
    """Make predictions on new data."""
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTOR - PREDICTION MODE")
    print("="*60)
    
    try:
        predictor = HeartDiseasePredictor(args.model, args.preprocessor)
        
        if args.file:
            # Load data from file
            import pandas as pd
            df = pd.read_csv(args.file) if args.file.endswith('.csv') else pd.read_excel(args.file)
            results = predictor.predict_batch(df)
            
            print(f"\nPredictions for {len(results)} samples:")
            for i, result in enumerate(results, 1):
                pred_label = 'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'
                print(f"\nSample {i}:")
                print(f"  Prediction: {pred_label}")
                if 'probability' in result:
                    print(f"  Confidence: {result['probability']:.2%}")
        else:
            print("\nPlease provide feature values to make a prediction.")
            print("Use --file to provide a CSV file with features, or")
            print("modify the code to provide features directly.")
            
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def explore_command(args):
    """Perform exploratory data analysis."""
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTOR - EXPLORATION MODE")
    print("="*60)
    
    try:
        # Load dataset
        df = load_dataset(args.dataset)
        
        # Perform EDA
        explorer = DataExplorer()
        explorer.explore(df, args.target)
        
    except Exception as e:
        print(f"\nError during exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Heart Disease Predictor - Supervised Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py train
  
  # Train with custom dataset
  python main.py train --dataset data/my_dataset.csv
  
  # Explore data
  python main.py explore
  
  # Make predictions
  python main.py predict --file data/new_samples.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--dataset', type=str, default=None,
                             help='Dataset filename (default: auto-detect in data/)')
    train_parser.add_argument('--target', type=str, default=None,
                             help='Target column name (default: auto-detect)')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Test set size (default: 0.2)')
    train_parser.add_argument('--random-state', type=int, default=42,
                             help='Random seed (default: 42)')
    train_parser.add_argument('--cv', type=int, default=5,
                             help='Cross-validation folds (default: 5)')
    train_parser.add_argument('--save-all', action='store_true',
                             help='Save all models, not just the best one')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--file', type=str,
                               help='CSV file with features to predict')
    predict_parser.add_argument('--model', type=str, default=None,
                               help='Path to model file (default: models/best_model.joblib)')
    predict_parser.add_argument('--preprocessor', type=str, default=None,
                               help='Path to preprocessor file (default: models/preprocessor.joblib)')
    
    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Perform EDA')
    explore_parser.add_argument('--dataset', type=str, default=None,
                               help='Dataset filename (default: auto-detect in data/)')
    explore_parser.add_argument('--target', type=str, default=None,
                               help='Target column name (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'explore':
        explore_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

