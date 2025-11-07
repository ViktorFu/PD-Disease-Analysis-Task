import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from tab_transformer_pytorch import TabTransformer
from scipy.special import softmax
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning from scikit-learn, which can be noisy with KernelExplainer
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- NEW: Model Wrapper for DeepExplainer ---
# This wrapper makes the model compatible with SHAP's DeepExplainer by accepting a single
# concatenated tensor and splitting it internally into categorical and continuous parts.
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, cat_feature_count):
        super().__init__()
        self.model = model
        self.cat_feature_count = cat_feature_count

    def forward(self, x):
        # x is a single tensor where categorical and continuous features are concatenated.
        # We split them back into two tensors here.
        x_categ = x[:, :self.cat_feature_count].long()
        x_cont = x[:, self.cat_feature_count:]
        return self.model(x_categ, x_cont)

SEED = 42

def compute_global_importance(shap_values):
    """
    Compute global feature importance (mean |SHAP| per feature) from SHAP output.
    Supports both list of arrays (multi-class) and single 3D array formats.
    """
    if isinstance(shap_values, list): # Multi-class output from KernelExplainer
        sv = np.array([np.abs(s) for s in shap_values]) # (C, N, F)
        global_imp = np.mean(sv, axis=(0, 1))
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: # Regression/other
        sv = np.abs(shap_values) # (N, F, C)
        global_imp = np.mean(sv, axis=(0, 2))
    else:
        raise ValueError("Unsupported shap_values format")
    return global_imp

def analyze_feature_importance():
    """
    Load a trained TabTransformer, run SHAP KernelExplainer, and save plots:
      1) SHAP bar plot (top 20)
      2) Correlation heatmap of those top 20 features
    """
    # --- 1. Paths ---
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / 'TabTransformer-based_Classification'
    data_dir = model_dir / 'data'
    
    model_path = model_dir / 'tab_transformer_model.pt'
    preprocessors_path = data_dir / 'preprocessors.joblib'
    # Use original, unscaled data for interpretable correlation heatmap and SHAP
    original_data_path = project_root / 'data' / 'imputed_train_val_set.csv'

    # Output paths
    output_dir = model_dir
    bar_plot_path = output_dir / 'shap_summary_bar_plot.png'
    heatmap_path = output_dir / 'top_20_features_correlation_heatmap.png'

    # --- 2. Load model, data, and preprocessors ---
    print("--- Loading model, data, and preprocessors ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        preprocessors = joblib.load(preprocessors_path)
        df_original = pd.read_csv(original_data_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure training and preprocessing have been run.")
        return

    # --- 3. Re-initialize model ---
    model_config = checkpoint['config']
    model = TabTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    
    # --- 4. Prepare data for SHAP ---
    print("\n--- Preparing data for SHAP analysis ---")
    cat_features = preprocessors['categorical_features']
    num_features = preprocessors['numerical_features']
    feature_names = cat_features + num_features
    
    # --- NEW: Wrap the model for SHAP ---
    wrapped_model = ModelWrapper(model, len(cat_features))

    # We use the original (unscaled, un-encoded) data and preprocess on the fly
    # This makes the SHAP plot more interpretable
    X = df_original[feature_names]
    
    # --- 5. Create a prediction function for SHAP KernelExplainer ---
    def predict_proba(x):
        # x is a numpy array from SHAP of shape (n_samples, n_features)
        df = pd.DataFrame(x, columns=feature_names)
        
        # Preprocess the data just like in preprocessing.py
        # Categorical
        x_categ_df = pd.DataFrame()
        for col, le in preprocessors['label_encoders'].items():
            # Handle unseen values by mapping them to a known category (e.g., the first one)
            str_series = df[col].astype(str)
            known_classes = le.classes_.astype(str)
            x_categ_df[col] = str_series.apply(lambda v: v if v in known_classes else known_classes[0])
            x_categ_df[col] = le.transform(x_categ_df[col])

        # Numerical
        x_cont_df = df[num_features]
        x_cont_scaled = preprocessors['scaler'].transform(x_cont_df)
        
        # Convert to tensors
        x_categ = torch.tensor(x_categ_df.values, dtype=torch.int64).to(device)
        x_cont = torch.tensor(x_cont_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(x_categ, x_cont)
        
        # Return probabilities for SHAP
        return softmax(logits.cpu().numpy(), axis=1)

    # --- 6. Compute SHAP values with KernelExplainer ---
    print("\n--- Calculating SHAP values on 100 samples (KernelExplainer can be slow) ---")
    # KernelExplainer is slower, so we use a smaller sample for explanation
    X_background = shap.sample(X, 100, random_state=SEED)
    X_eval = shap.sample(X, 100, random_state=SEED)

    explainer = shap.KernelExplainer(predict_proba, X_background)
    shap_values = explainer.shap_values(X_eval, nsamples=50) # nsamples for perturbation

    # --- 7. Bar plot (native SHAP) ---
    print("\n--- Saving SHAP summary bar plot ---")
    plt.figure()
    class_names_for_plot = [f"Class {i}" for i in range(len(shap_values))]
    shap.summary_plot(
        shap_values,
        X_eval,
        plot_type="bar",
        class_names=class_names_for_plot,
        show=False,
        max_display=20
    )
    plt.title("Top 20 Feature Importance (SHAP Bar Plot)")
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    plt.close()
    print(f"Saved bar plot to {bar_plot_path}")

    # --- 9. Global importance for correlation heatmap ---
    print("\n--- Computing global feature importance vector ---")
    global_importance = compute_global_importance(shap_values)

    feature_importance_df = (
        pd.DataFrame({'feature': feature_names, 'importance': global_importance})
        .sort_values('importance', ascending=False)
    )

    top_20_features = feature_importance_df.head(20)['feature'].tolist()

    # --- 10. Correlation heatmap ---
    print("\n--- Saving correlation heatmap for top 20 features ---")
    corr_matrix = df_original[top_20_features].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5
    )
    plt.title('Correlation Heatmap of Top 20 Most Important Features (Original Data)')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    print("\n--- SHAP Analysis Completed ---")


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    analyze_feature_importance()
