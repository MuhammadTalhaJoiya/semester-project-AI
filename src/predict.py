"""
Prediction module for making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from preprocessor import DataPreprocessor


class HeartDiseasePredictor:
    """Makes predictions using trained model."""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path (str, optional): Path to saved model
            preprocessor_path (str, optional): Path to saved preprocessor
        """
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "models" / "best_model.joblib"
        
        if preprocessor_path is None:
            project_root = Path(__file__).parent.parent
            preprocessor_path = project_root / "models" / "preprocessor.joblib"
        
        # Load model
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', 'Unknown')
            print(f"Model loaded: {self.model_name}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train a model first using train_model.py"
            )
        
        # Load preprocessor
        try:
            preprocessor_data = joblib.load(preprocessor_path)
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data.get('label_encoders', {})
            self.feature_columns = preprocessor_data['feature_columns']
            print(f"Preprocessor loaded with {len(self.feature_columns)} features")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Preprocessor file not found at {preprocessor_path}. "
                "Please train a model first to generate preprocessor."
            )
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data using saved preprocessor.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select only required columns in correct order
        df = df[self.feature_columns]
        
        # Encode categorical variables if needed
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # If unseen category, use most common class
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )
        
        return df_scaled
    
    def predict(self, input_data, return_proba=False):
        """
        Make prediction on input data.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            return_proba (bool): If True, return probability scores
            
        Returns:
            int or tuple: Prediction (and probabilities if return_proba=True)
        """
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            return prediction, probabilities
        else:
            return prediction
    
    def predict_batch(self, input_data_list):
        """
        Make predictions on multiple samples.
        
        Args:
            input_data_list (list): List of dictionaries or DataFrame
            
        Returns:
            list: List of predictions
        """
        if isinstance(input_data_list, pd.DataFrame):
            df = input_data_list
        else:
            df = pd.DataFrame(input_data_list)
        
        # Preprocess
        X = self.preprocess_input(df)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'prediction': pred,
                    'probability': probabilities[i][pred],
                    'probabilities': probabilities[i].tolist()
                })
            return results
        else:
            return [{'prediction': pred} for pred in predictions]


def predict_from_dict(features_dict):
    """
    Convenience function to make prediction from feature dictionary.
    
    Args:
        features_dict (dict): Dictionary of feature values
        
    Returns:
        dict: Prediction result
    """
    predictor = HeartDiseasePredictor()
    prediction, probabilities = predictor.predict(features_dict, return_proba=True)
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': float(probabilities[prediction]),
        'probabilities': {
            'No Heart Disease': float(probabilities[0]),
            'Heart Disease': float(probabilities[1])
        }
    }
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Heart Disease Predictor")
    print("="*50)
    
    try:
        predictor = HeartDiseasePredictor()
        
        # Example: You would provide actual feature values here
        # This is just a template - replace with actual values from your dataset
        print("\nExample prediction (using dummy data):")
        print("Note: Replace with actual feature values from your dataset")
        
        # The features should match your dataset columns
        # This is a placeholder - adjust based on your actual dataset
        example_features = {
            # Add your actual feature names and values here
            # Example: 'age': 63, 'sex': 1, 'cp': 3, etc.
        }
        
        if example_features:
            result = predict_from_dict(example_features)
            print(f"\nPrediction: {result['prediction_label']}")
            print(f"Confidence: {result['probability']:.2%}")
            print(f"\nProbabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.2%}")
        else:
            print("\nPlease provide feature values to make a prediction.")
            print("Use the predict_from_dict() function with your feature dictionary.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train a model first by running:")
        print("  python src/train_model.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

