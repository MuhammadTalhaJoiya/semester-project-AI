"""
Data preprocessing module for heart disease dataset.
Handles missing values, encoding, scaling, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor.
        
        Args:
            test_size (float): Proportion of test set (default: 0.2)
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def identify_target_column(self, df):
        """
        Automatically identify target column.
        Looks for common target column names.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            str: Name of target column
        """
        # Common target column names
        target_candidates = [
            'target', 'Target', 'TARGET',
            'label', 'Label', 'LABEL',
            'class', 'Class', 'CLASS',
            'heart_disease', 'HeartDisease', 'heartdisease',
            'disease', 'Disease', 'DISEASE',
            'output', 'Output', 'OUTPUT',
            'y', 'Y'
        ]
        
        for col in target_candidates:
            if col in df.columns:
                return col
        
        # If not found, assume last column is target
        return df.columns[-1]
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            print(f"\nHandling missing values:")
            print(missing[missing > 0])
            
            # For numeric columns, use mean imputation
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            
            # For categorical columns, use mode imputation
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            print("\nNo missing values found.")
        
        return df_clean
    
    def encode_categorical(self, df, target_col):
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df_encoded.columns if col != target_col]
        
        # Encode categorical features
        for col in feature_cols:
            if df_encoded[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def preprocess(self, df, target_column=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str, optional): Name of target column. If None, auto-detect.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*50)
        print("Starting Data Preprocessing")
        print("="*50)
        
        # Identify target column
        if target_column is None:
            target_column = self.identify_target_column(df)
        
        self.target_column = target_column
        print(f"\nTarget column identified: {target_column}")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_clean, target_column)
        
        # Separate features and target
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"\nFeatures: {len(self.feature_columns)}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("Preprocessing completed!")
        print("="*50)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath=None):
        """Save preprocessor objects for later use."""
        if filepath is None:
            project_root = Path(__file__).parent.parent
            filepath = project_root / "models" / "preprocessor.joblib"
        
        # Create models directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath=None):
        """Load preprocessor objects."""
        if filepath is None:
            project_root = Path(__file__).parent.parent
            filepath = project_root / "models" / "preprocessor.joblib"
        
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_dataset
    
    try:
        df = load_dataset()
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        print(f"\nPreprocessed data shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        print(f"Error: {e}")

