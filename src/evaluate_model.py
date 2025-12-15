"""
Model evaluation module.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from pathlib import Path


class ModelEvaluator:
    """Evaluates models and generates comprehensive reports."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate a single model and return metrics.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                pass
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, models_dict, X_test, y_test):
        """
        Evaluate multiple models.
        
        Args:
            models_dict (dict): Dictionary of {name: model}
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            pd.DataFrame: Comparison of all models
        """
        results_list = []
        
        for name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results_list.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(results_list)
        return comparison_df
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            metrics (dict): Evaluation metrics from evaluate_model
            save_path (str, optional): Path to save the plot
        """
        cm = metrics['confusion_matrix']
        model_name = metrics['model_name']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, metrics, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            metrics (dict): Evaluation metrics from evaluate_model
            save_path (str, optional): Path to save the plot
        """
        if metrics['y_pred_proba'] is None:
            print(f"Cannot plot ROC curve for {metrics['model_name']}: predict_proba not available")
            return
        
        y_test = metrics['y_test']
        y_pred_proba = metrics['y_pred_proba']
        model_name = metrics['model_name']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, metrics, save_path=None):
        """
        Generate text classification report.
        
        Args:
            metrics (dict): Evaluation metrics from evaluate_model
            save_path (str, optional): Path to save the report
        """
        model_name = metrics['model_name']
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        
        report = f"""
{'='*60}
Classification Report - {model_name}
{'='*60}

Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1_score']:.4f}

{'-'*60}
Detailed Classification Report:
{'-'*60}
{classification_report(y_test, y_pred, zero_division=0)}

{'='*60}
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
    
    def generate_all_plots(self, models_dict, X_test, y_test, output_dir=None):
        """
        Generate all evaluation plots for multiple models.
        
        Args:
            models_dict (dict): Dictionary of {name: model}
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            output_dir (str, optional): Directory to save plots
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "models"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            
            # Confusion matrix
            cm_path = output_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
            self.plot_confusion_matrix(metrics, cm_path)
            
            # ROC curve
            roc_path = output_dir / f"{name.lower().replace(' ', '_')}_roc_curve.png"
            self.plot_roc_curve(metrics, roc_path)
            
            # Text report
            report_path = output_dir / f"{name.lower().replace(' ', '_')}_report.txt"
            self.generate_report(metrics, report_path)


if __name__ == "__main__":
    # Test the evaluator
    from data_loader import load_dataset
    from preprocessor import DataPreprocessor
    from train_model import ModelTrainer
    
    try:
        print("Loading and preprocessing data...")
        df = load_dataset()
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        print("\nTraining models...")
        trainer = ModelTrainer()
        trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        print("\nEvaluating models...")
        evaluator = ModelEvaluator()
        comparison = evaluator.evaluate_all_models(trained_models, X_test, y_test)
        print("\nModel Comparison:")
        print(comparison)
        
        print("\nGenerating plots and reports...")
        evaluator.generate_all_plots(trained_models, X_test, y_test)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

