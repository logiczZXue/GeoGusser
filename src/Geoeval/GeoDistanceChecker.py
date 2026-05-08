import os
from geopy.distance import geodesic

def read_coordinates(file_path):
    """读取txt文件中的经纬度"""
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        lat, lon = map(float, line.split(','))
    return (lat, lon)

def calculate_distance_metrics(groundtruth_folder, evaluation_folder):
    """计算距离小于1KM、小于25KM、小于750KM的比例"""
    less_than_1km = 0
    less_than_25km = 0
    less_than_750km = 0
    total_count = 0

    for gt_file in os.listdir(groundtruth_folder):
        gt_file_path = os.path.join(groundtruth_folder, gt_file)
        eval_file_path = os.path.join(evaluation_folder, gt_file)

        if os.path.isfile(gt_file_path) and os.path.isfile(eval_file_path):
            gt_coords = read_coordinates(gt_file_path)
            eval_coords = read_coordinates(eval_file_path)

            # 计算距离
            distance = geodesic(gt_coords, eval_coords).kilometers

            # 更新统计
            total_count += 1
            if distance < 1:
                less_than_1km += 1
            if distance < 25:
                less_than_25km += 1
            if distance < 750:
                less_than_750km += 1

    if total_count == 0:
        print("No matching files found.")
        return

    # 计算比例
    less_than_1km_ratio = less_than_1km / total_count
    less_than_25km_ratio = less_than_25km / total_count
    less_than_750km_ratio = less_than_750km / total_count

    print(f"Total Files Evaluated: {total_count}")
    print(f"Proportion of distances < 1KM: {less_than_1km_ratio:.4f}")
    print(f"Proportion of distances < 25KM: {less_than_25km_ratio:.4f}")
    print(f"Proportion of distances < 750KM: {less_than_750km_ratio:.4f}")

# 示例使用
if __name__ == "__main__":
    groundtruth_folder = "data/im2gps3k_latlng"
    evaluation_folder = "data/llama_im2gps3k_latlng"
    calculate_distance_metrics(groundtruth_folder, evaluation_folder)
