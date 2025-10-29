import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json
from pathlib import Path

class SmartHierarchicalImputer(BaseEstimator, TransformerMixin):
    """
    A robust, production-grade Scikit-Learn compatible imputer that applies a 3x3 
    hierarchical strategy based on feature type and missing rate.
    """
    def __init__(self, high_thresh=0.3, low_thresh=0.03, features_json_path=None):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.features_json_path = features_json_path

    def fit(self, X, y=None):
        print("[FIT] Learning imputation strategy...")
        with open(self.features_json_path, 'r') as f:
            feature_types = json.load(f)
        
        numerical_features = feature_types.get('numerical_features', [])
        categorical_features = feature_types.get('binary_features', []) + feature_types.get('ordinal_features', [])
        
        self.feature_cols_ = [col for col in X.columns if col in numerical_features + categorical_features]
        missing_rate = X[self.feature_cols_].isnull().mean()

        def classify(features):
            high = [f for f in features if missing_rate.get(f, 0) > self.high_thresh]
            medium = [f for f in features if self.low_thresh < missing_rate.get(f, 0) <= self.high_thresh]
            low = [f for f in features if 0 < missing_rate.get(f, 0) <= self.low_thresh]
            return high, medium, low

        self.groups_ = {}
        self.groups_['high_num'], self.groups_['med_num'], self.groups_['low_num'] = classify(numerical_features)
        self.groups_['high_cat'], self.groups_['med_cat'], self.groups_['low_cat'] = classify(categorical_features)
        
        # Print classification results
        print("\n=== Feature Classification & Imputation Strategy ===")
        print(f"\nNumerical Features:")
        print(f"  - High missing (>30%): {self.groups_['high_num']} -> Median + Indicator")
        print(f"  - Medium missing (3%-30%): {self.groups_['med_num']} -> IterativeImputer")
        print(f"  - Low missing (0-3%): {self.groups_['low_num']} -> Median")
        
        print(f"\nCategorical Features (Binary + Ordinal):")
        print(f"  - High missing (>30%): {self.groups_['high_cat']} -> Constant (-1) + Indicator")
        print(f"  - Medium missing (3%-30%): {self.groups_['med_cat']} -> Constant (-1)")
        print(f"  - Low missing (0-3%): {self.groups_['low_cat']} -> Most Frequent")
        print("=" * 50)
        
        # Check for -1 in categorical features before using constant imputation
        if self.groups_['high_cat'] or self.groups_['med_cat']:
            all_high_med_cat = self.groups_['high_cat'] + self.groups_['med_cat']
            for col in all_high_med_cat:
                if col in X.columns and X[col].dtype != 'object':
                    unique_vals = X[col].dropna().unique()
                    if -1 in unique_vals:
                        raise ValueError(f"Feature '{col}' already contains value -1, cannot use -1 as imputation constant. Please check the data.")
        
        self.imputers_ = {}
        if self.groups_['high_num']:
            self.imputers_['high_num'] = SimpleImputer(strategy='median', add_indicator=True).fit(X[self.groups_['high_num']])
        if self.groups_['high_cat']:
            self.imputers_['high_cat'] = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True).fit(X[self.groups_['high_cat']])
        
        if self.groups_['low_num']:
            self.imputers_['low_num'] = SimpleImputer(strategy='median').fit(X[self.groups_['low_num']])
        
        # Med_cat uses constant -1, low_cat uses most_frequent
        if self.groups_['med_cat']:
            self.imputers_['med_cat'] = SimpleImputer(strategy='constant', fill_value=-1).fit(X[self.groups_['med_cat']])
        if self.groups_['low_cat']:
            self.imputers_['low_cat'] = SimpleImputer(strategy='most_frequent').fit(X[self.groups_['low_cat']])
        
        if self.groups_['med_num']:
            X_temp = X.copy()
            if self.groups_['low_num']:
                X_temp[self.groups_['low_num']] = self.imputers_['low_num'].transform(X[self.groups_['low_num']])
            
            # Handle med_cat and low_cat separately
            if self.groups_['med_cat']:
                X_temp[self.groups_['med_cat']] = self.imputers_['med_cat'].transform(X[self.groups_['med_cat']])
            if self.groups_['low_cat']:
                X_temp[self.groups_['low_cat']] = self.imputers_['low_cat'].transform(X[self.groups_['low_cat']])
            
            all_filled_cat = self.groups_['med_cat'] + self.groups_['low_cat']
            impute_cols = self.groups_['med_num'] + self.groups_['low_num'] + all_filled_cat
            self.imputers_['med_num'] = IterativeImputer(random_state=42, max_iter=30).fit(X_temp[impute_cols])

        # Store final column order
        original_cols = list(X.columns)
        indicator_cols = []
        
        # Determine which columns will be dropped and which indicators will be added
        high_missing_cols = self.groups_['high_num'] + self.groups_['high_cat']
        final_cols = [col for col in original_cols if col not in high_missing_cols]
        
        if 'high_num' in self.imputers_:
            final_cols += self.imputers_['high_num'].get_feature_names_out(self.groups_['high_num']).tolist()
        if 'high_cat' in self.imputers_:
            final_cols += self.imputers_['high_cat'].get_feature_names_out(self.groups_['high_cat']).tolist()

        self.final_columns_ = final_cols
        return self

    def transform(self, X):
        print("[TRANSFORM] Applying imputation...")
        X_transformed = X.copy()
        
        def transform_with_indicator(imputer, group_cols):
            data = imputer.transform(X_transformed[group_cols])
            new_cols = imputer.get_feature_names_out(group_cols)
            return pd.DataFrame(data, columns=new_cols, index=X_transformed.index)

        if 'high_num' in self.imputers_:
            df_high_num = transform_with_indicator(self.imputers_['high_num'], self.groups_['high_num'])
            X_transformed.drop(columns=self.groups_['high_num'], inplace=True)
            X_transformed = pd.concat([X_transformed, df_high_num], axis=1)

        if 'high_cat' in self.imputers_:
            df_high_cat = transform_with_indicator(self.imputers_['high_cat'], self.groups_['high_cat'])
            X_transformed.drop(columns=self.groups_['high_cat'], inplace=True)
            X_transformed = pd.concat([X_transformed, df_high_cat], axis=1)

        if 'low_num' in self.imputers_:
            X_transformed[self.groups_['low_num']] = self.imputers_['low_num'].transform(X[self.groups_['low_num']])
        
        if 'med_cat' in self.imputers_:
            X_transformed[self.groups_['med_cat']] = self.imputers_['med_cat'].transform(X[self.groups_['med_cat']])
        
        if 'low_cat' in self.imputers_:
            X_transformed[self.groups_['low_cat']] = self.imputers_['low_cat'].transform(X[self.groups_['low_cat']])
        
        if 'med_num' in self.imputers_:
            all_filled_cat = self.groups_['med_cat'] + self.groups_['low_cat']
            impute_cols = self.groups_['med_num'] + self.groups_['low_num'] + all_filled_cat
            imputed_data = self.imputers_['med_num'].transform(X_transformed[impute_cols])
            imputed_df = pd.DataFrame(imputed_data, columns=impute_cols, index=X_transformed.index)
            X_transformed[self.groups_['med_num']] = imputed_df[self.groups_['med_num']]

        # Reorder columns to match the order learned during fit
        return X_transformed[self.final_columns_]

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    
    train_file = data_dir / 'train_val_set.csv'
    test_file = data_dir / 'final_test_set.csv'
    features_file = data_dir / 'analysis_features.json'
    imputer_bundle_file = project_root / 'processing' / 'imputer.joblib'
    
    print("--- Loading Data ---")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    id_cols = ['PATNO', 'EVENT_ID']
    target_col = 'COHORT'
    feature_cols = [col for col in train_df.columns if col not in id_cols + [target_col]]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\n--- Initializing and Fitting the Imputer ---")
    imputer = SmartHierarchicalImputer(features_json_path=features_file)
    imputer.fit(X_train)

    print(f"\n--- Saving the fitted imputer to {imputer_bundle_file} ---")
    joblib.dump(imputer, imputer_bundle_file)

    print("\n--- Transforming Datasets ---")
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Reconstruct final dataframes
    final_train_df = pd.concat([train_df[id_cols], y_train, X_train_imputed], axis=1)
    final_test_df = pd.concat([test_df[id_cols], y_test, X_test_imputed], axis=1)
    
    print(f"\nTrain data shape after imputation: {final_train_df.shape}")
    print(f"Test data shape after imputation: {final_test_df.shape}")
    
    # Save the imputed datasets
    output_train_path = data_dir / 'imputed_train_val_set.csv'
    output_test_path = data_dir / 'imputed_test_set.csv'
    
    final_train_df.to_csv(output_train_path, index=False)
    final_test_df.to_csv(output_test_path, index=False)

    print(f"\nImputed training set saved to: {output_train_path}")
    print(f"Imputed test set saved to: {output_test_path}")
    print("\n--- Imputation Step Completed ---")
