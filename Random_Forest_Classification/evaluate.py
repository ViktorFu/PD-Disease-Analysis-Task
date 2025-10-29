import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, title, output_path, metrics_text):
    """
    This function prints and plots the confusion matrix and adds evaluation metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    # Add evaluation metrics text below the plot
    fig.text(0.5, -0.05, metrics_text, ha='center', wrap=True, fontsize=10, family='monospace')

    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {output_path}")
    plt.close(fig)

def evaluate_model():
    """
    Loads a trained model and evaluates its performance on the test set with detailed metrics.
    """
    # --- Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'imputed_test_set.csv'
    output_dir = project_root / 'Random_Forest_Classification'
    model_path = output_dir / 'random_forest_model.joblib'
    cm_plot_path = output_dir / 'evaluation_metrics.png'

    # --- Load Model ---
    print(f"--- Loading model from {model_path} ---")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return
    print("Model loaded successfully.")

    # --- Load Test Data ---
    print(f"\n--- Loading test data from {data_path} ---")
    df_test = pd.read_csv(data_path)

    # --- Prepare Data ---
    print("\n--- Preparing Test Data ---")
    target_col = 'COHORT' 
    model_features = model.feature_names_in_
    X_test = df_test[model_features]
    y_test = df_test[target_col]
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # --- Make Predictions ---
    print("\n--- Making Predictions on Test Set ---")
    y_pred = model.predict(X_test)

    # --- Evaluate Performance ---
    print("\n--- Evaluating Model Performance ---")
    target_names = [f"Cohort {i}" for i in sorted(y_test.unique())]
    
    # 1. Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 2. Classification Report
    print("\nClassification Report (Test Set):")
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    print(report_str)
    
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # 3. Format metrics for plotting
    metrics_summary = f"Overall Accuracy: {accuracy:.2%}\n\n"
    metrics_summary += "{:<12} {:<12} {:<12} {:<12}\n".format("Class", "Precision", "Recall", "F1-Score")
    metrics_summary += "-"*48 + "\n"
    for class_name, metrics in report_dict.items():
        if class_name in target_names:
            p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1-score']
            metrics_summary += "{:<12} {:<12.2f} {:<12.2f} {:<12.2f}\n".format(class_name, p, r, f1)

    # 4. Confusion Matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(cm, classes=target_names, title='Evaluation Metrics', 
                          output_path=cm_plot_path, metrics_text=metrics_summary)
    
    print("\n--- Evaluation Script Completed ---")

if __name__ == '__main__':
    evaluate_model()
