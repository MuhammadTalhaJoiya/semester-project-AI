"""
Model training module for heart disease prediction.
Implements multiple classification algorithms and selects the best model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import time


class ModelTrainer:
    """Trains multiple classification models and selects the best one."""
    
    def __init__(self, random_state=42):
        """
        Initialize model trainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classification models."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            )
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test, cv=5):
        """
        Train all models and evaluate them.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n" + "="*50)
        print("Training Models")
        print("="*50)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Evaluate on test set
            test_score = model.score(X_test, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            training_time = time.time() - start_time
            
            self.model_scores[name] = {
                'test_accuracy': test_score,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time
            }
            
            results.append({
                'Model': name,
                'Test Accuracy': f"{test_score:.4f}",
                'CV Mean': f"{cv_mean:.4f}",
                'CV Std': f"{cv_std:.4f}",
                'Training Time (s)': f"{training_time:.2f}"
            })
            
            print(f"  Test Accuracy: {test_score:.4f}")
            print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print(f"  Training Time: {training_time:.2f}s")
        
        # Display results summary
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Select best model based on test accuracy
        self.best_model_name = max(
            self.model_scores.keys(),
            key=lambda x: self.model_scores[x]['test_accuracy']
        )
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test Accuracy: {self.model_scores[self.best_model_name]['test_accuracy']:.4f}")
        
        return self.trained_models
    
    def save_best_model(self, filepath=None):
        """
        Save the best model to disk.
        
        Args:
            filepath (str, optional): Path to save model. If None, uses default.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train_all_models() first.")
        
        if filepath is None:
            project_root = Path(__file__).parent.parent
            filepath = project_root / "models" / "best_model.joblib"
        
        # Create models directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scores': self.model_scores[self.best_model_name]
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nBest model saved to: {filepath}")
    
    def save_all_models(self, directory=None):
        """
        Save all trained models to disk.
        
        Args:
            directory (str, optional): Directory to save models. If None, uses models/ folder.
        """
        if directory is None:
            project_root = Path(__file__).parent.parent
            directory = project_root / "models"
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.trained_models.items():
            # Clean filename
            filename = name.lower().replace(' ', '_') + '.joblib'
            filepath = directory / filename
            
            joblib.dump(model, filepath)
            print(f"Saved {name} to: {filepath}")
    
    @staticmethod
    def load_model(filepath=None):
        """
        Load a saved model.
        
        Args:
            filepath (str, optional): Path to model file. If None, loads best_model.joblib
            
        Returns:
            dict: Model data containing model, name, and scores
        """
        if filepath is None:
            project_root = Path(__file__).parent.parent
            filepath = project_root / "models" / "best_model.joblib"
        
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test the trainer
    from data_loader import load_dataset
    from preprocessor import DataPreprocessor
    
    try:
        print("Loading dataset...")
        df = load_dataset()
        
        print("\nPreprocessing data...")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        print("\nTraining models...")
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        print("\nSaving models...")
        trainer.save_best_model()
        trainer.save_all_models()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

