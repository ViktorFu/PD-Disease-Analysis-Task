import docx
import os
import re
import json
import sys

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the data directory
data_dir = os.path.join(script_dir, '..', 'data')

# Define absolute paths for input and output files
input_path = os.path.join(data_dir, 'Selected Features.docx')
text_output_path = os.path.join(data_dir, 'Selected_Features.txt')
json_output_path = os.path.join(data_dir, 'selected_features.json')

# Ensure the output directory exists
os.makedirs(data_dir, exist_ok=True)

# --- Step 1: DOCX to TXT ---
# Read the document
try:
    doc = docx.Document(input_path)
except Exception as e:
    print(f"Error reading docx file: {e}")
    sys.exit(1)

# Collect all content
content = []
# Extract all paragraphs
for p in doc.paragraphs:
    text = p.text.strip()
    if text:
        content.append(text)

# Save as a text file
try:
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    print(f"Successfully converted to text file: {text_output_path}")
    print(f"Extracted a total of {len(content)} lines of content")
except Exception as e:
    print(f"Error writing text file: {e}")
    sys.exit(1)

# --- Step 2: Extract features from TXT to JSON ---
print("\nStarting feature extraction from text file...")

categorized_features = {}
current_category = "General"  # Default category

# Regular expression to match feature lines, e.g., "lns – ..." or "DVS_LNS – ..."
# This expression captures the feature code before the '–' symbol
feature_pattern = re.compile(r'^([a-zA-Z0-9_]+)\s*–')

# Initialize the default category
categorized_features[current_category] = []

for line in content:
    match = feature_pattern.match(line)
    if match:
        # If it is a feature line, extract the feature name and add it to the current category
        feature_name = match.group(1).strip()
        if current_category not in categorized_features:
            categorized_features[current_category] = []
        categorized_features[current_category].append(feature_name)
    else:
        # If it's not a feature line, it's considered a new category title
        # Clean the title, remove emojis and leading spaces
        current_category = re.sub(r'^\W+\s*', '', line).strip()
        if current_category and current_category not in categorized_features:
             categorized_features[current_category] = []


# Remove empty categories
categorized_features = {k: v for k, v in categorized_features.items() if v}


# Save as a JSON file
try:
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_features, f, indent=4)
    print(f"Successfully extracted and saved features to JSON file: {json_output_path}")
    print(f"Total number of features: {sum(len(features) for features in categorized_features.values())}")
    print(f"Number of feature categories: {len(categorized_features)}")
except Exception as e:
    print(f"Error writing json file: {e}")

