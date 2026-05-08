import csv
import sys
import json
from collections import defaultdict
from tqdm import tqdm  # 进度条库
import os  # 用于获取当前文件夹路径

# 增加 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)

# 目标国家列表
target_countries = [
    "俄罗斯"
]

# 字典用于存储每个国家的 panoIDs
country_panoids = defaultdict(list)

# 输入和输出文件路径
input_file = 'tuxun_combined.csv'  # 替换为你的CSV文件路径
output_file = os.path.join(os.getcwd(), 'panoids.csv')  # 保存到当前文件夹

# 初始化目标国家的计数
remaining_countries = set(target_countries)

# 读取 CSV 文件并提取 panoIDs
with open(input_file, 'r', encoding='utf-8') as csvfile:
    # 获取文件行数以初始化进度条
    total_lines = sum(1 for _ in csvfile)
    csvfile.seek(0)  # 重置文件指针

    reader = csv.DictReader(csvfile)
    first_match_found = False  # 用于标记是否打印了第一个匹配的结果

    with tqdm(total=total_lines - 1, desc="Processing", unit="lines") as pbar:
        for row in reader:
            # 解析 data 字段为 JSON
            try:
                data = json.loads(row.get('data'))  # 替换 'data' 为实际字段名
                rounds = data.get('rounds', [])  # 从 JSON 中获取 rounds 列表
            except (json.JSONDecodeError, KeyError, TypeError):
                # 如果解析失败，跳过此行
                pbar.update(1)
                continue

            # 遍历 rounds 列表，提取目标信息
            for round_data in rounds:
                # 确保 round_data 是有效的字典
                if not isinstance(round_data, dict):
                    continue

                try:
                    country = round_data.get('nation')  # 获取国家信息
                    panoid = round_data.get('panoId')  # 获取 panoID
                except AttributeError:
                    continue

                # 打印第一个匹配结果
                if not first_match_found and country in target_countries:
                    print(f"第一个匹配的国家: {country}, panoID: {panoid}")
                    first_match_found = True

                # 检查是否是目标国家，并且该国家还未找到25个图片
                if country in target_countries and len(country_panoids[country]) < 160:
                    country_panoids[country].append(panoid)
                    
                    # 如果找到第60个，输出说明并移除国家
                    if len(country_panoids[country]) == 160:
                        print(f"国家 {country} 的 60 个 panoID 已经找齐。")
                        remaining_countries.discard(country)

                # 如果所有国家都已完成，提前退出外层循环
                if not remaining_countries:
                    break
            # 如果所有国家都已完成，提前退出外层循环
            if not remaining_countries:
                break
            pbar.update(1)

# 写入结果到新的 CSV 文件
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['nation', 'panoID'])  # 写入表头
    for country, panoids in country_panoids.items():
        for panoid in panoids:
            writer.writerow([country, panoid])

print(f"所有找到的 panoIDs 已写入到 {output_file}")
