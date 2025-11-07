import pandas as pd
import json
import os
import sys

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the data directory
data_dir = os.path.join(script_dir, '..', 'data')

# Define filenames
excel_filename = 'PPMI_Curated_Data_Cut_Public_20250321.xlsx'
csv_filename = 'PPMI_Curated_Data_Cut_Public_20250321.csv'
features_json_filename = 'selected_features.json'

# Define absolute paths for input and output files
input_path_xlsx = os.path.join(data_dir, excel_filename)
output_path_csv = os.path.join(data_dir, csv_filename)
json_path_features = os.path.join(data_dir, features_json_filename)

# Variable for comparison (used for log output)
output_filename_csv = csv_filename

print(f"Reading Excel file: {input_path_xlsx}")
try:
    df = pd.read_excel(input_path_xlsx)
    print("-> Successfully read Excel file.")
    print(f"   - Data shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Save as CSV
    print(f"\nSaving data to CSV file: {output_path_csv}")
    df.to_csv(output_path_csv, index=False)

    print("-> Conversion successful!")
    print(f"   - CSV file saved at: {output_path_csv}")

except FileNotFoundError:
    print(f"Error: Input Excel file not found at {input_path_xlsx}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during Excel to CSV conversion: {e}")
    sys.exit(1)

print("Verifying Features in CSV ---")

# 1. Load feature list from JSON file
print(f"Loading required features from: {json_path_features}")
try:
    with open(json_path_features, 'r', encoding='utf-8') as f:
        categorized_features = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON features file not found at {json_path_features}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_path_features}")
    sys.exit(1)

all_features_from_json = set()
for category, features in categorized_features.items():
    for feature in features:
        all_features_from_json.add(feature)

print(f"-> Successfully loaded {len(all_features_from_json)} unique features from {features_json_filename}.")

# 2. Get column names from the newly created CSV file
print(f"Getting column names from the newly created CSV: {output_filename_csv}")
csv_columns = set(df.columns)
print(f"-> Found {len(csv_columns)} columns in the CSV file.")

# 3. Compare
print("\nComparing JSON features against CSV columns...")
missing_features = all_features_from_json - csv_columns
found_count = len(all_features_from_json) - len(missing_features)

print("\n--- Verification Report ---")
print(f"Total features to check: {len(all_features_from_json)}")
print(f"Features found in CSV: {found_count}")

if missing_features:
    print(f"\nERROR: {len(missing_features)} features are missing from the CSV file!")
    print("List of missing features:")
    for feature in sorted(list(missing_features)):
        print(f"  - {feature}")
    sys.exit(1) # Exit with an error code
else:
    print("\n✅ SUCCESS: All required features from the JSON file are present in the CSV file.")
