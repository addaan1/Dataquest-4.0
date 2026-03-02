
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

def plot_error_analysis(y_true, y_pred, model_name, file_path='error_analysis.png'):
    """
    Plots predicted vs. actual values to analyze model errors.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Error Analysis for {model_name}")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Error analysis plot saved to {file_path}")

def plot_feature_importance(model, feature_names, model_name, file_path='feature_importance.png'):
    """
    Plots feature importance for a tree-based model.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top 20 Feature Importances for {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Feature importance plot saved to {file_path}")

def plot_learning_curve(estimator, X, y, model_name, file_path='learning_curve.png'):
    """
    Generates and plots a learning curve for a given model.
    """
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(.1, 1.0, 5),
        scoring="neg_root_mean_squared_error"
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Learning curve plot saved to {file_path}")
