# 提取GT中的关键信息输出到txt文件
# 输入 GT表格selected_panoids.csv
# 输出 包含地点国家大洲的txt文件，文件名与panoid一致
import pandas as pd
import os
import re

# 定义国家及其对应的大洲
country_to_continent = {
    "法国": "Europe",
    "俄罗斯": "Europe",
    "乌干达": "Africa",
    "美国": "North America",
    "澳大利亚": "Australia",
    "日本": "Asia",
    "加拿大": "North America",
    "巴西": "South America",
    "韩国": "Asia",
    "秘鲁": "South America",
    "英国": "Europe",
    "南非": "Africa",
    "泰国": "Asia",
    "芬兰": "Europe",
    "肯尼亚": "Africa",
    "意大利": "Europe",
    "新西兰": "Australia",
    "墨西哥": "North America",
    "印度": "Asia",
    "德国": "Europe"
}

# 读取 CSV 文件
csv_file_path = './data/GT.csv'
save_dir = './data/ground_truth'  # 保存目录
os.makedirs(save_dir, exist_ok=True)  # 创建保存目录（如果不存在）

def info2text(csv_file_path, save_dir):
    # 读取 CSV 文件
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ['panoID', 'description']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing required column: {column}")
            return

    # 遍历每一行
    for index, row in df.iterrows():
        try:
            panoid = row['panoID']
            description = row['description']

            # 使用正则表达式查找描述中的国家和大洲信息
            match = re.search(
                r'the image was most likely taken in ([^,]+), ([^,]+)(?:, ([^.]+))?\.$', 
                description, re.IGNORECASE
            )

            if match:
                location = match.group(1)  # Location
                country = match.group(2)   # Country
                continent = match.group(3)  # Continent (可能为 None)

                if not continent:
                    # 如果没有 CONTINENT，查找国家对应的大洲
                    continent = country_to_continent.get(country, "Unknown")

                output_content = f"{location}, {country}, {continent}"

                # 保存到文件
                os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
                output_file_path = os.path.join(save_dir, f"{panoid}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)
            else:
                print(f"Row {index}: Description does not match the expected format. Skipping.")
        except KeyError as e:
            print(f"Row {index}: Missing key {e}. Skipping.")
        except Exception as e:
            print(f"Row {index}: Error processing row: {e}. Skipping.")

    print("文件已成功保存到指定目录。")

def get_description(csv_file_path, save_dir):
    """从 CSV 文件中提取 description 列并保存为 panoid.txt 文件"""
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每一行
    for index, row in df.iterrows():
        panoid = row['panoID']
        description = row['description']

        # 保存 description 到文件
        output_file_path = os.path.join(save_dir, f"{panoid}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(description)

    print("所有描述已成功保存到指定目录。")

if __name__ == "__main__":
    info2text(csv_file_path, save_dir)
    # get_description(csv_file_path, './data/gt_description')
