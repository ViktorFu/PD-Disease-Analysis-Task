import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

SEED = 42

def compute_global_importance(shap_values):
    """
    Compute global feature importance (mean |SHAP| per feature) from SHAP output.

    Supports:
    - list of (n_samples, n_features) arrays  [classic multi-class]
    - (n_samples, n_features) array           [binary/regression]
    - (n_samples, n_features, n_classes) array [your current case]
    """
    sv = shap_values

    # Case 1: list of arrays, one per class
    if isinstance(sv, list):
        per_class_imps = []
        for arr in sv:
            # arr shape: (n_samples, n_features)
            imp = np.mean(np.abs(arr), axis=0)  # (n_features,)
            per_class_imps.append(imp)
        per_class_imps = np.vstack(per_class_imps)  # (n_classes, n_features)
        global_imp = per_class_imps.mean(axis=0)    # (n_features,)
        return global_imp

    sv = np.array(sv)

    # Case 2: 2D array -> (n_samples, n_features)
    if sv.ndim == 2:
        # binary / regression style
        global_imp = np.mean(np.abs(sv), axis=0)  # (n_features,)
        return global_imp

    # Case 3: 3D array -> (n_samples, n_features, n_classes)
    if sv.ndim == 3:
        # this is your case: (N, F, C)
        # average over samples (axis=0) and classes (axis=2)
        global_imp = np.mean(np.abs(sv), axis=(0, 2))  # (n_features,)
        return global_imp

    raise RuntimeError(
        f"Unhandled shap_values format with ndim={sv.ndim} and shape={sv.shape}"
    )


def analyze_feature_importance():
    """
    Load trained RF pipeline, run SHAP on a sample, and save:
      1) SHAP bar plot (top 20)
      2) correlation heatmap of those top 20
    """
    # --- Paths ---
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'imputed_train_val_set.csv'
    model_path = project_root / 'Random_Forest_Classification' / 'random_forest_model.joblib'
    output_dir = project_root / 'Random_Forest_Classification'
    bar_plot_path = output_dir / 'shap_summary_bar_plot.png'
    heatmap_path = output_dir / 'top_20_features_correlation_heatmap.png'

    # --- Load model and data ---
    print("--- Loading model and data ---")
    best_model_pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # --- Prepare data ---
    print("\n--- Preparing data for SHAP analysis ---")
    id_cols = ['PATNO', 'EVENT_ID']
    target_col = 'COHORT'

    X = df.drop(columns=id_cols + [target_col])
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=0)  # ensure SHAP won't choke on NaN

    rf_model = best_model_pipeline.named_steps["classifier"]

    # --- Sample for SHAP ---
    print("Sampling background and evaluation sets...")
    X_background = shap.sample(X, 100, random_state=SEED)
    X_eval = shap.sample(X, 1000, random_state=SEED)

    # --- Compute SHAP values ---
    print("\n--- Calculating SHAP values on 1000 samples ---")
    explainer = shap.TreeExplainer(rf_model, X_background)
    shap_values = explainer.shap_values(X_eval)  # shape likely (1000, 75, 4) here

    # --- Bar plot (native SHAP) ---
    print("\n--- Saving SHAP summary bar plot ---")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_eval,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title("Top 20 Feature Importance (SHAP Bar Plot) on 1000 Samples")
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    plt.close()
    print(f"Saved bar plot to {bar_plot_path}")

    # --- Global importance for correlation heatmap ---
    print("\n--- Computing global feature importance vector ---")
    global_importance = compute_global_importance(shap_values)

    if len(global_importance) != X_eval.shape[1]:
        raise RuntimeError(
            f"Global importance length {len(global_importance)} != "
            f"feature count {X_eval.shape[1]}"
        )

    feature_importance_df = (
        pd.DataFrame({
            'feature': X_eval.columns,
            'importance': global_importance
        })
        .sort_values('importance', ascending=False)
    )

    top_20_features = feature_importance_df.head(20)['feature'].tolist()

    # --- Correlation heatmap ---
    print("\n--- Saving correlation heatmap for top 20 features ---")
    corr_matrix = X_eval[top_20_features].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Correlation Heatmap of Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    print("\n--- SHAP Analysis Completed ---")


if __name__ == '__main__':
    np.random.seed(SEED)
    analyze_feature_importance()