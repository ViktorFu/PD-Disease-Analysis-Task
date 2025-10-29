import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_path, output_dir):
    """
    Splits the dataset into training/validation and test sets based on patients.

    Args:
        input_path (str): Path to the input CSV file (e.g., the output of process_main_dataset.py).
        output_dir (str): Directory to save the output files.
    """
    print(f"Reading data from {input_path}...")
    original_df = pd.read_csv(input_path)
    print("Data loaded successfully.")

    # Step 1: Create a "Patient-Label" mapping table
    print("Step 1: Creating patient-label mapping table...")
    patient_main_df = original_df[original_df['EVENT_ID'] == 'BL'][['PATNO', 'COHORT']].drop_duplicates()
    print(f"Found {len(patient_main_df)} unique patients at baseline.")

    print("\nClass dist (all baseline patients):")
    print(patient_main_df['COHORT'].value_counts())

    # Step 2: First split - separate out the "final test set"
    print("\nStep 2: Splitting patients into train/val and test sets...")
    train_val_patients, test_patients = train_test_split(
        patient_main_df,
        test_size=0.20,
        random_state=42,
        stratify=patient_main_df['COHORT']
    )

    print("\nClass dist (train_val patients):")
    print(train_val_patients['COHORT'].value_counts())

    print("\nClass dist (test patients):")
    print(test_patients['COHORT'].value_counts())

    print(f"\nTrain/Val patients: {len(train_val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    # Filter the original dataframe to create the train/val and test sets
    train_val_df = original_df[original_df['PATNO'].isin(train_val_patients['PATNO'])]
    test_df = original_df[original_df['PATNO'].isin(test_patients['PATNO'])]

    print(f"Total records in train/val set: {len(train_val_df)}")
    print(f"Total records in test set: {len(test_df)}")
    
    # Save the datasets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    train_val_path = os.path.join(output_dir, 'train_val_set.csv')
    test_path = os.path.join(output_dir, 'final_test_set.csv')

    train_val_df.to_csv(train_val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train/Val dataset saved to {train_val_path}")
    print(f"Final Test dataset saved to {test_path}")
    print("Splitting process completed.")

if __name__ == '__main__':
    # Assuming the script is in the 'processing' directory.
    # It reads raw data from 'data/' and outputs the split CSVs back into 'data/'.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'data', 'raw_data_3.0.csv')
    output_directory = os.path.join(project_root, 'data')

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please ensure the raw data file 'raw_data_3.0.csv' is in the 'data' directory.")
    else:
        split_dataset(input_file, output_directory)
