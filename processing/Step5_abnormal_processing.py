import pandas as pd
from pathlib import Path
from collections import Counter

def process_and_clean_abnormalities(input_path, output_path):
    """
    Identifies abnormal samples, removes entries that are outliers in multiple tests,
    and saves the cleaned dataset.

    An entry is considered for removal if its (PATNO, EVENT_ID) pair appears
    as an outlier in two or more different cognitive tests.

    Args:
        input_path (str or Path): The path to the input CSV file.
        output_path (str or Path): The path to save the cleaned CSV file.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {input_path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    # Define the columns and the number of top samples to consider as outliers
    columns_of_interest = {
        'hvlt_discrimination': 1,
        'hvlt_immediaterecall': 1,
        'HVLTREC': 2,  # Top 2
        'hvlt_retention': 1,
        'HVLTFPRL': 1
    }

    abnormal_ids = []
    print("\nIdentifying potential outliers from top values in specified columns:")
    for col, n_largest in columns_of_interest.items():
        if col in df.columns:
            top_samples = df.nlargest(n_largest, col)
            print(f"--- Top {n_largest} for '{col}' ---")
            for _, row in top_samples.iterrows():
                patno_event = (row['PATNO'], row['EVENT_ID'])
                print(f"  PATNO: {row['PATNO']}, EVENT_ID: {row['EVENT_ID']}, Value: {row[col]}")
                abnormal_ids.append(patno_event)
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")
    
    # Count how many times each (PATNO, EVENT_ID) pair appears in the outlier lists
    id_counts = Counter(abnormal_ids)
    
    # Identify pairs that appear in 2 or more outlier lists
    ids_to_remove = {patno_event for patno_event, count in id_counts.items() if count >= 2}
    
    if not ids_to_remove:
        print("\nNo entries identified as outliers in 2 or more tests. No rows will be removed.")
        return

    print(f"\nFound {len(ids_to_remove)} entries that are outliers in 2 or more tests. They will be removed:")
    for patno, event_id in sorted(list(ids_to_remove)):
        print(f"  - PATNO: {patno}, EVENT_ID: {event_id}")

    # Remove the identified rows
    original_rows = len(df)
    df_indexed = df.set_index(['PATNO', 'EVENT_ID'])
    cleaned_df = df_indexed.drop(index=list(ids_to_remove), errors='ignore').reset_index()
    
    # Save the cleaned dataframe
    cleaned_df.to_csv(output_path, index=False)
    
    print(f"\nRemoved {original_rows - len(cleaned_df)} rows.")
    print(f"Cleaned data saved to {output_path}. New shape: {cleaned_df.shape}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_file = project_root / 'data' / 'raw_data_2.0.csv'
    output_file = project_root / 'data' / 'raw_data_3.0.csv'
    process_and_clean_abnormalities(input_file, output_file)
