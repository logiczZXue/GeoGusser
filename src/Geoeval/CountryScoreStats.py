# 统计每个国家的平均分、前 10% 和后 10% 的平均分
# 输入 游戏记录tuxun_combined.csv
# 输出 结果result_final.csv
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict

def process_scores(csv_file_path, output_file_path, target_countries):
    country_scores = defaultdict(list)
    all_scores = []
    
    # 逐块读取 CSV 文件
    with pd.read_csv(
        csv_file_path, 
        chunksize=5000,  # 每次读取 5000 行
        usecols=['data'],  # 只读取必要的列
        dtype={'data': str}  # 指定数据类型
    ) as reader:
        for chunk in tqdm(reader, desc="Processing CSV", mininterval=1.0):
            for _, row in chunk.iterrows():
                try:
                    data = row['data']
                    json_data = json.loads(data)

                    # 从 player.roundResults 中提取分数
                    player = json_data.get("player", {})
                    round_results = player.get("roundResults", [])

                    rounds = json_data.get("rounds", [])
                    for i, round_info in enumerate(rounds):
                        if isinstance(round_info, dict):
                            country = round_info.get("nation", "未知")
                            if country in target_countries:
                                if i < len(round_results):
                                    score = round_results[i].get("score", 0)
                                    if isinstance(score, (int, float)) and score > 500:  # 仅保留 score > 500
                                        country_scores[country].append(score)
                                        all_scores.append(score)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {row['data']}")

    # 统计每个国家的平均分、前 10% 和后 10% 的平均分
    result = {
        "Country": [],
        "Overall Average": [],
        "Top 10% Average": [],
        "Bottom 10% Average": []
    }

    for country, scores in country_scores.items():
        if scores:
            scores.sort()
            n = len(scores)
            overall_avg = sum(scores) / n
            top_10_percent = scores[:max(1, n*10 // 100)]  # 前 10%
            bottom_10_percent = scores[-max(1, n*10 // 100):]  # 后 10%

            avg_top_10 = sum(top_10_percent) / len(top_10_percent) if top_10_percent else 0
            avg_bottom_10 = sum(bottom_10_percent) / len(bottom_10_percent) if bottom_10_percent else 0

            result["Country"].append(country)
            result["Overall Average"].append(overall_avg)
            result["Top 10% Average"].append(avg_top_10)
            result["Bottom 10% Average"].append(avg_bottom_10)

    if all_scores:
        all_scores.sort()
        n = len(all_scores)
        overall_avg = sum(all_scores) / n
        top_10_percent = all_scores[:max(1, n*10 // 100)]  # 前 10%
        bottom_10_percent = all_scores[-max(1, n*10 // 100):]  # 后 10%
        avg_top_10 = sum(top_10_percent) / len(top_10_percent) if top_10_percent else 0
        avg_bottom_10 = sum(bottom_10_percent) / len(bottom_10_percent) if bottom_10_percent else 0
    else:
        overall_avg = 0
        avg_top_10 = 0
        avg_bottom_10 = 0
    result["Country"].append("All")
    result["Overall Average"].append(overall_avg)
    result["Top 10% Average"].append(avg_top_10)
    result["Bottom 10% Average"].append(avg_bottom_10)
    # 输出到 CSV
    output_df = pd.DataFrame(result)
    output_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"统计结果已保存到文件：{output_file_path}")

if __name__ == "__main__":
    csv_file_path = "tuxun_combined.csv"  # 替换为你的 CSV 文件路径
    output_file_path = "result_final.csv"
    target_countries = {"芬兰", "加拿大", "美国", "中国", "德国", "法国", "日本", "印度", "巴西", "韩国"}
    process_scores(csv_file_path, output_file_path, target_countries)