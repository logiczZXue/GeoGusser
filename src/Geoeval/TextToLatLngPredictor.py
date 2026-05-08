import os
import requests
import openai
import urllib.parse
from tqdm import tqdm

# 使用 GPT 提取更细致的地址
def use_gpt_to_extract_address(description, gpt_api_key):
    openai.api_key = gpt_api_key
    prompt = (
        "Analyze the following text and predict the most likely location based on the description. "
        "The output should be a specific location such as a city, country, or region. "
        "If the description is vague or unclear, make a reasonable assumption based on the available details. "
        "Always provide a plausible address.\n\n"
        f"Description:\n{description}\n\n"
        "Output the predicted location (e.g., 'Kampala, Uganda' or 'East Africa'):"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # 调用 GPT-4o
        messages=[
            {"role": "system", "content": "You are a helpful assistant for extracting locations from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# 使用 Google Geocoding API 获取经纬度
def get_lat_lng_from_address(address, api_key):
    encoded_address = urllib.parse.quote(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        return None

# 处理文件夹中的所有文件
def process_folder_with_gpt(input_folder, output_folder, google_api_key, gpt_api_key):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in tqdm(os.listdir(input_folder),total=len(os.listdir(input_folder)),desc="Processing files"):
        if file_name.endswith('.txt'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            
            # 检查是否已经存在结果文件
            if os.path.exists(output_file_path):
                # print(f"{file_name} already processed. Skipping...")
                continue

            # 读取文件内容
            with open(input_file_path, 'r') as file:
                content = file.read().strip()
            
            # 调用 GPT 提取地址
            address = use_gpt_to_extract_address(content, gpt_api_key)
            if address:
                # print(f"Predicted address for {file_name}: {address}")
                # 获取经纬度
                lat_lng = get_lat_lng_from_address(address, google_api_key)
                if lat_lng:
                    # 写入结果到输出文件
                    with open(output_file_path, 'w') as out_file:
                        out_file.write(f"{lat_lng[0]}, {lat_lng[1]}")
                    # print(f"Processed {file_name}: {lat_lng[0]}, {lat_lng[1]}")
                else:
                    # print(f"Failed to get GPS for {file_name} with address: {address}")
                    # 写入默认结果，防止输出为空
                    with open(output_file_path, 'w') as out_file:
                        out_file.write("0.0, 0.0")  # 默认值
            else:
                # print(f"Failed to extract address for {file_name}")
                # 写入默认结果
                with open(output_file_path, 'w') as out_file:
                    out_file.write("0.0, 0.0")

# 示例调用
google_api_key = "AIzaSyA2gP66fP9yj26bt3JoVVdmv28wV1mdsoA"  # 替换为你的 Google API 密钥
gpt_api_key = "KKK"   # 替换原有的sk-proj-开头的密钥
input_folder = "data/llamas_im2gps3k"             # 输入文件夹路径
output_folder = "data/llama_im2gps3k_latlng"      # 输出文件夹路径
process_folder_with_gpt(input_folder, output_folder, google_api_key, gpt_api_key)
print(f"All files processed. Results saved to {output_folder}")
