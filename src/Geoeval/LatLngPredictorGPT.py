import os
import base64
import requests
import openai
import urllib.parse
from PIL import Image
from io import BytesIO

# 调整图像大小
def resize_image(image_path, size=(512, 512)):
    """调整图像大小并返回 BytesIO 对象"""
    try:
        with Image.open(image_path) as img:
            img = img.resize(size)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            return buffer
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

# 将图像编码为 Base64 字符串
def encode_image_to_base64(image_path):
    """将图像编码为 Base64 字符串"""
    resized_image = resize_image(image_path)
    if resized_image:
        return base64.b64encode(resized_image.read()).decode("utf-8")
    return None

# 使用 GPT 提取经纬度，确保输出为经纬度格式
# 增加精确度，预测结果的经纬度小数位尽可能多

def use_gpt_with_image(description, image_base64, gpt_api_key):
    openai.api_key = gpt_api_key
    prompt = (
        "You are an advanced AI assistant specialized in geolocation analysis. "
        "Based on a textual description of a location and an associated image encoded in Base64, Let's think step by step."
        "predict the most likely latitude and longitude in decimal format. Ensure the latitude and longitude "
        "are as accurate as possible with at least 5 decimal places, aiming for precision within 1 kilometer. "
        "If the details are ambiguous, use contextual clues to estimate a plausible latitude and longitude. "
        "Always respond strictly in the format:\n"
        "'Latitude, Longitude'. No additional text or explanation.\n\n"
        "Location description:\n"
        f"{description}\n\n"
        "Associated image (Base64 encoded):\n"
        f"{image_base64}\n"
    )
    try:
        while True:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a geolocation expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            result = response['choices'][0]['message']['content'].strip()
            # 验证是否符合纬度，经度格式
            if result.count(',') == 1:
                lat, lng = result.split(',')
                try:
                    float(lat.strip())
                    float(lng.strip())
                    return result
                except ValueError:
                    pass
            print("Invalid response format from GPT. Retrying...")
    except Exception as e:
        print(f"Error using GPT: {e}")
        return "0.0, 0.0"

# 使用 Google Geocoding API 获取经纬度
def get_lat_lng_from_address(address, api_key):
    try:
        encoded_address = urllib.parse.quote(address)
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"
        response = requests.get(url)
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    except Exception as e:
        print(f"Error fetching GPS for address {address}: {e}")
    return None

# 处理文件夹中的所有文件
def process_folder_with_gpt_and_images(input_folder, image_folder, output_folder, google_api_key, gpt_api_key):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # 检查是否已经存在结果文件
            if os.path.exists(output_file_path):
                print(f"{file_name} already processed. Skipping...")
                continue

            # 读取文件内容
            with open(input_file_path, 'r') as file:
                content = file.read().strip()

            # 构造图片路径
            panoid = file_name.split('.')[0]  # 假设文件名与 panoid 对应
            image_path = os.path.join(image_folder, f"{panoid}.jpg")

            if os.path.exists(image_path):
                # 编码图片
                image_base64 = encode_image_to_base64(image_path)

                if image_base64:
                    # 调用 GPT 提取经纬度
                    lat_lng = use_gpt_with_image(content, image_base64, gpt_api_key)
                    print(f"Predicted for {file_name}: {lat_lng}")

                    # 写入结果到输出文件
                    with open(output_file_path, 'w') as out_file:
                        out_file.write(lat_lng)
                else:
                    print(f"Failed to encode image for {file_name}")
                    with open(output_file_path, 'w') as out_file:
                        out_file.write("0.0, 0.0")
            else:
                print(f"Image not found for {file_name}, skipping...")
                with open(output_file_path, 'w') as out_file:
                    out_file.write("0.0, 0.0")

# 示例调用
google_api_key = "lll"  # 替换为你的 Google API 密钥
gpt_api_key = "kkk"   
input_folder = "data/im2gps3k_geos"      # 输入文件夹路径
image_folder = "data/im2gps3k"           # 图片文件夹路径
output_folder = "data/im2gps3k_geo_LAT"  # 输出文件夹路径

process_folder_with_gpt_and_images(input_folder, image_folder, output_folder, google_api_key, gpt_api_key)
print(f"All files processed. Results saved to {output_folder}")
