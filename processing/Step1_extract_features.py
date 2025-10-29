import docx
import os
import re
import json
import sys

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建到data目录的绝对路径
data_dir = os.path.join(script_dir, '..', 'data')

# 定义输入和输出文件的绝对路径
input_path = os.path.join(data_dir, 'Selected Features.docx')
text_output_path = os.path.join(data_dir, 'Selected_Features.txt')
json_output_path = os.path.join(data_dir, 'selected_features.json')

# 确保输出目录存在
os.makedirs(data_dir, exist_ok=True)

# --- 步骤 1: DOCX 到 TXT ---
# 读取文档
try:
    doc = docx.Document(input_path)
except Exception as e:
    print(f"Error reading docx file: {e}")
    sys.exit(1)

# 收集所有内容
content = []
# 提取所有段落
for p in doc.paragraphs:
    text = p.text.strip()
    if text:
        content.append(text)

# 保存为文本文件
try:
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    print(f"已成功转换为文本文件: {text_output_path}")
    print(f"共提取 {len(content)} 行内容")
except Exception as e:
    print(f"Error writing text file: {e}")
    sys.exit(1)

# --- 步骤 2: 从 TXT 提取特征到 JSON ---
print("\n开始从文本文件提取特征...")

categorized_features = {}
current_category = "General"  # 默认分类

# 正则表达式匹配特征行，例如 "lns – ..." 或 "DVS_LNS – ..."
# 这个表达式会捕获'–'符号前的特征代码
feature_pattern = re.compile(r'^([a-zA-Z0-9_]+)\s*–')

# 初始化默认分类
categorized_features[current_category] = []

for line in content:
    match = feature_pattern.match(line)
    if match:
        # 如果是特征行，提取特征名并添加到当前分类
        feature_name = match.group(1).strip()
        if current_category not in categorized_features:
            categorized_features[current_category] = []
        categorized_features[current_category].append(feature_name)
    else:
        # 如果不是特征行，则认为是新的分类标题
        # 清理标题，移除表情符号和前导空格
        current_category = re.sub(r'^\W+\s*', '', line).strip()
        if current_category and current_category not in categorized_features:
             categorized_features[current_category] = []


# 移除空的分类
categorized_features = {k: v for k, v in categorized_features.items() if v}


# 保存为JSON文件
try:
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_features, f, indent=4)
    print(f"已成功提取并保存特征到JSON文件: {json_output_path}")
    print(f"特征总数: {sum(len(features) for features in categorized_features.values())}")
    print(f"特征类别数: {len(categorized_features)}")
except Exception as e:
    print(f"Error writing json file: {e}")

