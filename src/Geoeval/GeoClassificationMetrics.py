import os
from sklearn.metrics import accuracy_score, recall_score, f1_score

def read_panoid_file(file_path):
    """读取PANOID.txt文件并返回地点、国家和大洲的列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()  # 读取第一行并去除空白字符
    print(f"Debug: {file_path} -> {line}")  # 打印读取的内容，便于调试

    # 改进分割逻辑，确保只分成三个部分
    values = line.rsplit(', ', maxsplit=2)  # 从右开始分割两次，确保最后两个部分是国家和大洲
    if len(values) != 3:  # 检查字段数量
        raise ValueError(f"Invalid format in file {file_path}: {line}")
    return values

def evaluate_directories(dir_a, dir_b):
    """评测目录A和B中的同名PANOID.txt文件，并分别统计每个字段的指标"""
    ground_truth_locations = []
    ground_truth_countries = []
    ground_truth_continents = []

    predictions_locations = []
    predictions_countries = []
    predictions_continents = []

    for filename in os.listdir(dir_a):
        file_a_path = os.path.join(dir_a, filename)
        file_b_path = os.path.join(dir_b, filename)

        if os.path.exists(file_b_path):
            try:
                # 读取 ground_truth 和 zero 文件内容
                a_location, a_country, a_continent = read_panoid_file(file_a_path)
                b_location, b_country, b_continent = read_panoid_file(file_b_path)

                # 存储每个字段的数据
                ground_truth_locations.append(a_location)
                ground_truth_countries.append(a_country)
                ground_truth_continents.append(a_continent)

                predictions_locations.append(b_location)
                predictions_countries.append(b_country)
                predictions_continents.append(b_continent)

                # 检测 `none` 值作为错误
                if b_location == "none":
                    print(f"Warning: None value in location for {filename}")
                if b_country == "none":
                    print(f"Warning: None value in country for {filename}")
                if b_continent == "none":
                    print(f"Warning: None value in continent for {filename}")

            except ValueError as e:
                print(f"Error processing files: {e}")
                continue  # 跳过当前文件，继续处理其他文件

    # 分别计算每个字段的指标
    try:
        location_accuracy = accuracy_score(ground_truth_locations, predictions_locations)
        location_recall = recall_score(ground_truth_locations, predictions_locations, average='macro', zero_division=0)
        location_f1 = f1_score(ground_truth_locations, predictions_locations, average='macro', zero_division=0)

        country_accuracy = accuracy_score(ground_truth_countries, predictions_countries)
        country_recall = recall_score(ground_truth_countries, predictions_countries, average='macro', zero_division=0)
        country_f1 = f1_score(ground_truth_countries, predictions_countries, average='macro', zero_division=0)

        continent_accuracy = accuracy_score(ground_truth_continents, predictions_continents)
        continent_recall = recall_score(ground_truth_continents, predictions_continents, average='macro', zero_division=0)
        continent_f1 = f1_score(ground_truth_continents, predictions_continents, average='macro', zero_division=0)
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        return None

    # 返回每个字段的指标
    return {
        'location': {
            'accuracy': round(location_accuracy, 3),
            'recall': round(location_recall, 3),
            'f1_score': round(location_f1, 3),
        },
        'country': {
            'accuracy': round(country_accuracy, 3),
            'recall': round(country_recall, 3),
            'f1_score': round(country_f1, 3),
        },
        'continent': {
            'accuracy': round(continent_accuracy, 3),
            'recall': round(continent_recall, 3),
            'f1_score': round(continent_f1, 3),
        },
    }

# 使用示例
dir_a = 'data/ground_truth'
dir_b = 'data/llama'
evaluation_results = evaluate_directories(dir_a, dir_b)
print(evaluation_results)
