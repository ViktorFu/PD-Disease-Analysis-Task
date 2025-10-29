import pandas as pd
import json
import os
import sys

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the data directory
data_dir = os.path.join(script_dir, '..', 'data')

# Define filenames
csv_filename = 'PPMI_Curated_Data_Cut_Public_20250321.csv'
features_json_filename = 'selected_features.json'

# Define absolute paths for input and output files
input_path_csv = os.path.join(data_dir, csv_filename)
json_path_features = os.path.join(data_dir, features_json_filename)

# Load CSV data
print(f"Loading CSV file: {input_path_csv}")
try:
    df = pd.read_csv(input_path_csv, low_memory=False)
    print(f"-> Successfully read CSV file. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {input_path_csv}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Load feature list
print(f"Loading features from: {json_path_features}")
try:
    with open(json_path_features, 'r', encoding='utf-8') as f:
        categorized_features = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON features file not found at {json_path_features}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_path_features}")
    sys.exit(1)

# Extract all features from the categorized features
all_features_from_json = set()
for category, features in categorized_features.items():
    for feature in features:
        all_features_from_json.add(feature)

print(f"-> Successfully loaded {len(all_features_from_json)} unique features.")
print(f"   Features: {sorted(list(all_features_from_json))}")

print("\n--- Step 3: Creating raw_data.csv with Selected Features and Identifiers ---")
try:
    # Convert set to list
    feature_list = list(all_features_from_json)
    
    # Remove COHORT from the feature list to move it to the end later
    # The verification step has already ensured that 'COHORT' exists in the list
    feature_list.remove('COHORT')
    
    # Build the final column order: identifiers first, 56 features in the middle, target at the end
    ordered_columns = ['PATNO', 'EVENT_ID'] + feature_list + ['COHORT']

    # Select and reorder columns from the original DataFrame
    df_raw_data = df[ordered_columns].copy()
    
    # Define the output path
    raw_data_path = os.path.join(data_dir, 'raw_data_1.0.csv')
    
    # Save the new CSV file
    print(f"Saving {len(ordered_columns)} columns (56 features + PATNO, EVENT_ID, COHORT) to: {raw_data_path}")
    df_raw_data.to_csv(raw_data_path, index=False)
    
    print("-> Successfully created raw_data_1.0.csv.")
    print(f"   - Shape of raw_data_1.0.csv: {df_raw_data.shape}")
    print(f"   - Identifier columns 'PATNO', 'EVENT_ID' are included at the start.")
    print(f"   - Target column 'COHORT' is the last column.")

except Exception as e:
    print(f"\nAn error occurred while creating raw_data_1.0.csv: {e}")
    sys.exit(1)

print("\n--- Script finished ---")