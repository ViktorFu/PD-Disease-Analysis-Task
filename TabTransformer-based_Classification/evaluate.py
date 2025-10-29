import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from tab_transformer_pytorch import TabTransformer

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
    Loads a trained TabTransformer model and evaluates its performance on the test set.
    """
    # --- Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / 'TabTransformer-based_Classification'
    data_dir = output_dir / 'data'
    model_path = output_dir / 'tab_transformer_model.pt'
    data_path = data_dir / 'test_processed.pt'
    preprocessors_path = data_dir / 'preprocessors.joblib'
    cm_plot_path = output_dir / 'evaluation_metrics.png'

    # --- Setup Device ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    print(f"--- Loading model from {model_path} ---")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return

    model_config = checkpoint.get('config')
    if not model_config:
        print("Error: Model configuration not found in checkpoint.")
        return
        
    model = TabTransformer(**model_config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- Load Test Data & Preprocessors ---
    print(f"\n--- Loading test data from {data_path} ---")
    test_data = torch.load(data_path)
    test_dataset = TensorDataset(test_data['x_categ'], test_data['x_cont'], test_data['y'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    preprocessors = joblib.load(preprocessors_path)

    # --- Make Predictions ---
    print("\n--- Making Predictions on Test Set ---")
    y_pred_list, y_test_list = [], []
    
    with torch.no_grad():
        for x_categ, x_cont, y in test_loader:
            x_categ, x_cont = x_categ.to(DEVICE), x_cont.to(DEVICE)
            outputs = model(x_categ, x_cont)
            preds = torch.argmax(outputs, dim=1)
            y_pred_list.extend(preds.cpu().numpy())
            y_test_list.extend(y.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_test = np.array(y_test_list)

    # --- Evaluate Performance ---
    print("\n--- Evaluating Model Performance ---")
    target_encoder = preprocessors['target_encoder']
    target_names = [f"Cohort {c}" for c in target_encoder.classes_]
    
    # 1. Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 2. Classification Report
    print("\nClassification Report (Test Set):")
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report_str)
    
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # 3. Format metrics for plotting
    metrics_summary = f"Overall Accuracy: {accuracy:.2%}\n\n"
    metrics_summary += "{:<12} {:<12} {:<12} {:<12}\n".format("Class", "Precision", "Recall", "F1-Score")
    metrics_summary += "-"*52 + "\n"
    for class_name, metrics in report_dict.items():
        if class_name in target_names:
            p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1-score']
            metrics_summary += "{:<12} {:<12.2f} {:<12.2f} {:<12.2f}\n".format(class_name, p, r, f1)

    # 4. Confusion Matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(cm, classes=target_names, title='TabTransformer Evaluation Metrics', 
                          output_path=cm_plot_path, metrics_text=metrics_summary)
    
    print("\n--- Evaluation Script Completed ---")

if __name__ == '__main__':
    evaluate_model()
