import pandas as pd
import json
import os
import sys

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建到data目录的绝对路径
data_dir = os.path.join(script_dir, '..', 'data')

# 定义文件名
csv_filename = 'PPMI_Curated_Data_Cut_Public_20250321.csv'
features_json_filename = 'selected_features.json'

# 定义输入和输出文件的绝对路径
input_path_csv = os.path.join(data_dir, csv_filename)
json_path_features = os.path.join(data_dir, features_json_filename)

# 加载 CSV 数据
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

# 加载特征列表
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

# 从分类特征中提取所有特征
all_features_from_json = set()
for category, features in categorized_features.items():
    for feature in features:
        all_features_from_json.add(feature)

print(f"-> Successfully loaded {len(all_features_from_json)} unique features.")
print(f"   Features: {sorted(list(all_features_from_json))}")

print("\n--- Step 3: Creating raw_data.csv with Selected Features and Identifiers ---")
try:
    # 从集合转换为列表
    feature_list = list(all_features_from_json)
    
    # 从特征列表中移除 COHORT，以便稍后将其移动到末尾
    # 验证步骤已确保 'COHORT' 存在于列表中
    feature_list.remove('COHORT')
    
    # 构建最终的列顺序：标识符在前，56个特征居中，目标在后
    ordered_columns = ['PATNO', 'EVENT_ID'] + feature_list + ['COHORT']

    # 从原始DataFrame中选择并排序列
    df_raw_data = df[ordered_columns].copy()
    
    # 定义输出路径
    raw_data_path = os.path.join(data_dir, 'raw_data_1.0.csv')
    
    # 保存新的CSV文件
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