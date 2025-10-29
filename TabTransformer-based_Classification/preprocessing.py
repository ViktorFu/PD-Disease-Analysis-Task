import pandas as pd
import numpy as np
import json
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

def preprocess_data():
    """
    Loads raw imputed data, applies robust preprocessing, and saves the processed 
    data, preprocessors, and metadata for full reproducibility.
    """
    # --- Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    output_dir = project_root / 'TabTransformer-based_Classification' / 'data'
    output_dir.mkdir(exist_ok=True)

    train_file = data_dir / 'imputed_train_val_set.csv'
    test_file = data_dir / 'imputed_test_set.csv'
    features_file = data_dir / 'analysis_features.json'

    # --- Load Data and Feature Definitions ---
    print("--- Loading data and feature definitions ---")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    with open(features_file, 'r') as f:
        feature_types = json.load(f)

    categorical_features = feature_types.get('binary_features', []) + feature_types.get('ordinal_features', [])
    numerical_features = feature_types.get('numerical_features', [])
    
    # --- MODEL-SPECIFIC OVERRIDE ---
    print("\n--- Applying model-specific feature classification override ---")
    features_to_move = ['cogstate', 'NP1DPRS']
    categorical_features = [f for f in categorical_features if f not in features_to_move]
    numerical_features.extend(features_to_move)
    print(f"Moved {features_to_move} from ordinal to numerical for this model.")

    # --- SEPARATE INDICATOR COLUMNS ---
    indicator_cols = [col for col in train_df.columns if col.startswith('missing_')]
    # Ensure numerical_features only contains true continuous variables for scaling
    numerical_features = [f for f in numerical_features if f not in indicator_cols]
    all_numerical_features = numerical_features + indicator_cols # Keep track of the final order
    
    target_col = 'COHORT'
    id_cols = ['PATNO', 'EVENT_ID']
    
    # --- ROBUSTNESS CHECK 1: Prevent ID Leakage ---
    leak_cols = set(id_cols) & (set(categorical_features) | set(all_numerical_features))
    assert len(leak_cols) == 0, f"ID columns leaked into model input: {leak_cols}"

    # --- Preprocessing ---
    print("\n--- Starting preprocessing ---")
    
    # 1. Label Encode Target Variable
    target_encoder = LabelEncoder()
    train_df[target_col] = target_encoder.fit_transform(train_df[target_col])
    test_df[target_col] = target_encoder.transform(test_df[target_col])
    
    # 2. Process Categorical Features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    print(f"Encoded {len(categorical_features)} categorical features.")

    # 3. Process Numerical Features (ONLY true continuous ones)
    # --- ROBUSTNESS CHECK 2: Ensure no NaNs are present before scaling ---
    assert train_df[numerical_features].notna().all().all(), "NaNs in continuous features (train)."
    assert test_df[numerical_features].notna().all().all(), "NaNs in continuous features (test)."
    
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])
    print(f"Standardized {len(numerical_features)} continuous features. Kept {len(indicator_cols)} indicators as raw 0/1.")

    # --- Prepare Tensors ---
    print("\n--- Converting data to PyTorch Tensors ---")
    
    def df_to_tensors(df):
        x_categ = torch.tensor(df[categorical_features].values, dtype=torch.int64)
        # Note: all_numerical_features now defines the correct final order
        x_cont = torch.tensor(df[all_numerical_features].values, dtype=torch.float32)
        y = torch.tensor(df[target_col].values, dtype=torch.int64)
        return x_categ, x_cont, y

    x_categ_train, x_cont_train, y_train = df_to_tensors(train_df)
    x_categ_test, x_cont_test, y_test = df_to_tensors(test_df)

    # --- Save Processed Data and Preprocessors with Rich Metadata ---
    print("\n--- Saving processed data and preprocessors ---")

    torch.save({'x_categ': x_categ_train, 'x_cont': x_cont_train, 'y': y_train}, output_dir / 'train_processed.pt')
    torch.save({'x_categ': x_categ_test, 'x_cont': x_cont_test, 'y': y_test}, output_dir / 'test_processed.pt')

    joblib.dump({
        'target_encoder': target_encoder,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'categorical_features': categorical_features,
        'numerical_features': all_numerical_features,
        'target_col': target_col,
        'id_cols': id_cols,
        'moved_from_ordinal_to_numerical': features_to_move
    }, output_dir / 'preprocessors.joblib')
    
    print(f"Processed files and rich metadata saved in: {output_dir}")
    print("\n--- Preprocessing script completed successfully! ---")

if __name__ == '__main__':
    preprocess_data()
