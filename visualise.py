import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_feature_importance(model, feature_names, output_file=None):
    """Plot feature importance from trained model"""
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=[feature_names[i] for i in indices])
    plt.title("Feature Importance")
    plt.tight_layout()
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
    return plt

def plot_correlation_matrix(df, output_file=None):
    """Plot correlation matrix for numeric features"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
    return plt
