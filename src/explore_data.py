"""
Exploratory Data Analysis (EDA) module.
Generates visualizations and statistics for the heart disease dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataExplorer:
    """Performs exploratory data analysis on the dataset."""
    
    def __init__(self, output_dir=None):
        """
        Initialize data explorer.
        
        Args:
            output_dir (str, optional): Directory to save plots
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "models"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def explore(self, df, target_column=None):
        """
        Perform complete EDA on dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str, optional): Name of target column
        """
        print("\n" + "="*60)
        print("Exploratory Data Analysis")
        print("="*60)
        
        # Identify target column
        if target_column is None:
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
                    target_column = col
                    break
            
            if target_column is None:
                target_column = df.columns[-1]
        
        print(f"\nTarget column: {target_column}")
        
        # Basic statistics
        self.print_basic_stats(df, target_column)
        
        # Generate visualizations
        self.plot_target_distribution(df, target_column)
        self.plot_feature_distributions(df, target_column)
        self.plot_correlation_heatmap(df, target_column)
        self.plot_feature_target_relationships(df, target_column)
        
        print("\n" + "="*60)
        print("EDA Complete! All plots saved to:", self.output_dir)
        print("="*60)
    
    def print_basic_stats(self, df, target_column):
        """Print basic statistics about the dataset."""
        print("\n" + "-"*60)
        print("Dataset Information")
        print("-"*60)
        print(f"Shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Samples: {df.shape[0]}")
        
        print(f"\nColumn Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values")
        
        print(f"\nTarget Distribution:")
        print(df[target_column].value_counts())
        print(f"\nTarget Distribution (%):")
        print(df[target_column].value_counts(normalize=True) * 100)
        
        print(f"\nNumerical Features Summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
    
    def plot_target_distribution(self, df, target_column):
        """Plot target variable distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        target_counts = df[target_column].value_counts()
        axes[0].bar(target_counts.index, target_counts.values, color=['skyblue', 'salmon'])
        axes[0].set_title('Target Variable Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(target_counts.index)
        axes[0].set_xticklabels(['No Disease', 'Disease'])
        for i, v in enumerate(target_counts.values):
            axes[0].text(target_counts.index[i], v, str(v), ha='center', va='bottom')
        
        # Pie chart
        axes[1].pie(target_counts.values, labels=['No Disease', 'Disease'], 
                   autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
        axes[1].set_title('Target Variable Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / "target_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path.name}")
        plt.close()
    
    def plot_feature_distributions(self, df, target_column):
        """Plot distributions of all features."""
        feature_cols = [col for col in df.columns if col != target_column]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("\nNo numerical features to plot.")
            return
        
        # Calculate grid size
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram with target overlay
                df[df[target_column] == 0][col].hist(alpha=0.5, label='No Disease', 
                                                      bins=20, ax=axes[i], color='skyblue')
                df[df[target_column] == 1][col].hist(alpha=0.5, label='Disease', 
                                                     bins=20, ax=axes[i], color='salmon')
                axes[i].set_title(f'{col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / "feature_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")
        plt.close()
    
    def plot_correlation_heatmap(self, df, target_column):
        """Plot correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("\nNot enough numerical features for correlation heatmap.")
            return
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")
        plt.close()
        
        # Print correlation with target
        if target_column in numeric_df.columns:
            print(f"\nCorrelation with Target ({target_column}):")
            correlations = numeric_df.corr()[target_column].sort_values(ascending=False)
            correlations = correlations[correlations.index != target_column]
            for feature, corr in correlations.items():
                print(f"  {feature}: {corr:.3f}")
    
    def plot_feature_target_relationships(self, df, target_column):
        """Plot relationships between features and target."""
        feature_cols = [col for col in df.columns if col != target_column]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return
        
        # Box plots for top correlated features
        top_n = min(6, len(numeric_cols))
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_column in numeric_df.columns:
            correlations = numeric_df.corr()[target_column].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != target_column]
            top_features = correlations.head(top_n).index.tolist()
        else:
            top_features = numeric_cols[:top_n].tolist()
        
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(top_features):
            if i < len(axes):
                df.boxplot(column=col, by=target_column, ax=axes[i], grid=False)
                axes[i].set_title(f'{col} by Target', fontweight='bold')
                axes[i].set_xlabel('Target')
                axes[i].set_ylabel(col)
                axes[i].set_xticklabels(['No Disease', 'Disease'])
        
        # Hide extra subplots
        for i in range(len(top_features), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Feature-Target Relationships', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / "feature_target_relationships.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")
        plt.close()


if __name__ == "__main__":
    # Test the explorer
    from data_loader import load_dataset
    
    try:
        print("Loading dataset...")
        df = load_dataset()
        
        print("\nPerforming EDA...")
        explorer = DataExplorer()
        explorer.explore(df)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

