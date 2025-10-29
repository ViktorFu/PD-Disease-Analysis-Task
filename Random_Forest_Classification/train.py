import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pathlib import Path
import time

# --- NEW IMPORTS ---
# We need the Pipeline from imblearn, not sklearn, to work with SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
# Import metrics to add a final report
from sklearn.metrics import classification_report

def hyperparameter_tuning():
    """
    Performs hyperparameter tuning for a Random Forest classifier using GridSearchCV,
    integrating SMOTE within a pipeline to handle class imbalance, and saves the best model.
    """
    # --- Define Paths --- (No changes here)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'imputed_train_val_set.csv'
    output_dir = project_root / 'Random_Forest_Classification'
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'random_forest_model.joblib'

    # --- Load Data --- (No changes here)
    print("--- Loading Data ---")
    df = pd.read_csv(data_path)
    
    # --- Prepare Data --- (No changes here)
    print("\n--- Preparing Data for Training ---")
    id_cols = ['PATNO', 'EVENT_ID']
    target_col = 'COHORT'
    X = df.drop(columns=id_cols + [target_col])
    y = df[target_col]
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # --- Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Starting Hyperparameter Tuning with GridSearchCV ---")
    start_time = time.time()
    
    # --- MODIFICATION 1: Create an imblearn Pipeline ---
    # This pipeline defines the steps: first apply SMOTE, then train the classifier.
    # This is the key to correctly using SMOTE with cross-validation.
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # --- MODIFICATION 2: Update the parameter grid for the pipeline ---
    # Parameter names must be prefixed with the step name ('classifier__')
    # --- MODIFICATION 1: A more constrained parameter grid to fight overfitting ---
    param_grid = {
        # Keep n_estimators reasonable. More trees can sometimes increase overfitting.
        'classifier__n_estimators': [100, 200],
        
        # This is the MOST IMPORTANT parameter to tune.
        # Force the model to use simpler, less deep trees.
        'classifier__max_depth': [5, 8, 10, 12], 
        
        # Force leaves to be larger, preventing the model from creating tiny leaves
        # just to fit a few synthetic samples.
        'classifier__min_samples_leaf': [5, 10, 15],
        
        # Similar to above, require more samples to even consider a split.
        'classifier__min_samples_split': [10, 20, 30],
        
        'classifier__max_features': ['sqrt'], # 'sqrt' is often a good default

        # --- MODIFICATION 2 (Optional but Recommended): Tune SMOTE's neighbors ---
        # A smaller k creates more localized (potentially overfitting) samples.
        # A larger k creates more global, diverse samples. Let's test it.
        'smote__k_neighbors': [3, 5, 7]
    }
    
    # Initialize cross-validation (No changes here)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Setup GridSearchCV to use the new pipeline
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                               scoring='f1_macro', n_jobs=-1, cv=cv, verbose=2)
    
    # Run the grid search (No changes here)
    grid_search.fit(X, y)
    
    end_time = time.time()
    print(f"\nGrid search completed in {(end_time - start_time) / 60:.2f} minutes.")
    
    # --- Print Best Results --- (No changes here)
    print("\n--- Best Hyperparameters Found ---")
    print(grid_search.best_params_)
    print(f"\nBest F1 Macro Score (from CV): {grid_search.best_score_:.4f}")
    
    # The best_estimator_ is now the entire fitted pipeline (SMOTE + Classifier)
    best_model_pipeline = grid_search.best_estimator_

    # --- NEW: Add a final report on the training data for immediate feedback ---
    print("\n--- Final Model Performance Report (on full training data) ---")
    y_pred = best_model_pipeline.predict(X)
    print(classification_report(y, y_pred))

    # --- Save The Best Model ---
    print(f"\n--- Saving The Best Model to {model_path} ---")
    joblib.dump(best_model_pipeline, model_path)
    print("Best model saved successfully.")
    
    print("\n--- Hyperparameter Tuning Script Completed ---")

if __name__ == '__main__':
    hyperparameter_tuning()