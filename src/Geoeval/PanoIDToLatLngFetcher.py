import os
import requests
import time
import random

# 从文件夹提取所有 .txt 文件的文件名（去掉扩展名）
def extract_panoids_from_folder(folder_path):
    panoid_list = []
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                # 去掉 .txt 后缀，获取文件名作为 panoid
                panoid = os.path.splitext(file_name)[0]
                panoid_list.append(panoid)
    except Exception as e:
        print(f"Error while accessing folder: {e}")
    return panoid_list

# 查询单个 panoid 的经纬度
def get_lat_lng(panoid, api_key):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?pano={panoid}&key={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") == "OK":
            lat = data["location"]["lat"]
            lng = data["location"]["lng"]
            return lat, lng
        else:
            print(f"Error: {data.get('status')} for panoid {panoid}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# 处理文件夹中的 panoid 列表并查询经纬度
def process_panoids_from_folder(folder_path, api_key):
    panoid_list = extract_panoids_from_folder(folder_path)
    if not panoid_list:
        print("No valid .txt files found in the folder.")
        return []

    results = []
    for panoid in panoid_list:
        lat_lng = get_lat_lng(panoid, api_key)
        if lat_lng:
            results.append((panoid, lat_lng))
            print(f"PanoID: {panoid} -> Latitude: {lat_lng[0]}, Longitude: {lat_lng[1]}")
        else:
            print(f"Failed to fetch data for PanoID: {panoid}")

        # 加入随机 sleep，模拟人工操作
        sleep_time = random.uniform(1, 3)  # 随机延迟 1 到 3 秒
        print(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    return results

# 示例
folder_path = "data/ground_truth"  # 替换为你的文件夹路径
api_key = "lll"  # 替换为你的 Google Maps API Key
results = process_panoids_from_folder(folder_path, api_key)

# 输出结果到文件或打印
output_file = "data/results.txt"
with open(output_file, "w") as f:
    for panoid, lat_lng in results:
        f.write(f"PanoID: {panoid}, Latitude: {lat_lng[0]}, Longitude: {lat_lng[1]}\n")

print(f"Results saved to {output_file}")
