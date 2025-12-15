"""
Data loading module for heart disease dataset.
Automatically detects and loads dataset from data/ folder.
"""

import os
import pandas as pd
from pathlib import Path


def load_dataset(filename=None):
    """
    Load dataset from data/ folder.
    
    Args:
        filename (str, optional): Specific filename to load. If None, tries common names.
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If dataset file is not found
    """
    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Common dataset filenames to try
    common_names = [
        "heart_disease.csv",
        "heart.csv",
        "dataset.csv",
        "heart_disease.xlsx",
        "heart.xlsx"
    ]
    
    if filename:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    else:
        # Try to find dataset automatically
        file_path = None
        for name in common_names:
            potential_path = data_dir / name
            if potential_path.exists():
                file_path = potential_path
                break
        
        if file_path is None:
            available_files = list(data_dir.glob("*"))
            raise FileNotFoundError(
                f"No dataset file found in {data_dir}. "
                f"Please place your dataset (CSV or Excel) in the data/ folder. "
                f"Available files: {[f.name for f in available_files]}"
            )
    
    # Load dataset based on extension
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Dataset loaded successfully from: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    return df


if __name__ == "__main__":
    # Test the data loader
    try:
        df = load_dataset()
        print("\nDataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")

