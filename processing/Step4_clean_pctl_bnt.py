import pandas as pd
import numpy as np
import os

def preprocess_pctl_bnt(value):
    """
    Cleans and converts the PCTL_BNT column to a numerical format.
    - Handles range strings (e.g., "41 to 59") by taking the midpoint.
    - Handles '<1' by converting it to 0.
    - Converts numerical strings to floats.
    - Keeps NaN values as is.
    """
    # First, check for NaN or None and return it directly. A robust way is to use pd.isna().
    if pd.isna(value):
        return np.nan

    # Convert to string for consistent processing.
    s_value = str(value).strip()

    if ' to ' in s_value:
        try:
            low, high = s_value.split(' to ')
            return (float(low) + float(high)) / 2
        except (ValueError, TypeError):
            return np.nan # Return NaN if splitting or conversion fails.
    elif s_value == '<1':
        # The user's code block returns 0. Note: the user's summary table suggested 0.5,
        # but we follow the provided code logic.
        return 0.0
    else:
        try:
            # Try converting directly to a number (handles "99", "2", "1", etc.).
            return float(s_value)
        except ValueError:
            # If all attempts fail, return NaN.
            return np.nan

def main():
    """
    Main function to load the raw data, process the PCTL_BNT column,
    and save the cleaned data to a new file.
    """
    # Construct paths relative to the script's location for robustness.
    # The script is in processing/, so the parent directory is the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, 'data', 'raw_data_1.0.csv')
    output_path = os.path.join(project_root, 'data', 'raw_data_2.0.csv')

    print(f"Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    print("\n--- Before Processing ---")
    print("Original 'PCTL_BNT' value counts:")
    print(df['PCTL_BNT'].value_counts(dropna=False).sort_index())

    print("\nProcessing 'PCTL_BNT' column...")
    
    # Apply the cleaning function, replacing the original column
    df['PCTL_BNT'] = df['PCTL_BNT'].apply(preprocess_pctl_bnt)
    
    print("\n--- After Processing ---")
    print("Cleaned 'PCTL_BNT' data type:", df['PCTL_BNT'].dtype)
    print("Cleaned 'PCTL_BNT' value counts:")
    print(df['PCTL_BNT'].value_counts(dropna=False).sort_index())
    
    print(f"\nSaving processed data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Processing complete. Cleaned file saved successfully.")


if __name__ == "__main__":
    main()
