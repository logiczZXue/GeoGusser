import openai
from PIL import Image
import base64
import os
from io import BytesIO

# 设置 OpenAI API 密钥
openai.api_key = ""

def resize_image(image_path, size=(512, 512)):
    """调整图像大小"""
    with Image.open(image_path) as img:
        img = img.resize(size)
        return img

def encode_image_to_base64(image_path):
    """将图像编码为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

def process_image(image_path):
    """处理单张图片并与 GPT 进行一轮对话"""
    # 编码图像
    encoded_image = encode_image_to_base64(image_path)

    # 提取图片文件的前半部分命名（不包括后缀和east等后缀）
    image_prefix = os.path.basename(image_path).replace("_east.jpg", "")

    # 初始化对话消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant for analyzing images."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
    ]

    questions =  [#"Q1: Are there significant geographical features (mountains, trees, soil types, etc.) that can help determine the region?",
                  #"Q2: Are there local or culturally/historically significant buildings, structures, or signs in a specific language that indicate a country?",
                  "Q1: Are there easily identifiable street elements, such as road direction (left or right), types of bollards, utility poles, or license plate colors?",
                  #"Q4: Are there recognizable signs, fire hydrants, guideposts, or other indicators that can help pinpoint the country and city?",
                  "Q2: Are there identifiable sidewalk patterns (tile color, shape, arrangement) or clothing styles that can help determine the city?",
                  """Let's think step by step. Based on the question I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph (note: it must be a paragraph).\
                    For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate, most likely located in Southeast Asia. The signs are in Thai and the hydrant is a green circle, indicating that it is in Thailand. Right-side traffic, cylindrical bollards with blue markings and license plates with black lettering on a white background meet Thai standards. Traditional Thai architecture, such as pitched roofs and wooden structures, further points to a specific location in Thailand. Square gray sidewalk tiles and non-motorized lanes marked with red asphalt are specific urban design features that help narrow down the location. Combining tropical vegetation, Thai-language road signs, traditional architecture, and specific urban design features, this image was most likely taken in a city in Bangkok, Thailand, Asia."
                  """    
    ]

    for question in questions:
        messages.append({"role": "user", "content": question})
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1280
        )
        reply = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
    
    # 获取最后一轮的总结
    final_summary = messages[-1]["content"]
    messages = [
        {
            "role": "system",
            "content": (
            "Extract the city, country, and continent from the result."
            "Output them strictly in the order: city, country, continent."
            "Separate them with a comma and a space."
            "If any of these is not found, write 'none' in its place."
            "The continent must be one of the six continents excluding Antarctica (Asia, Africa, North America, South America, Europe, Oceania), and no other continents should be listed. If uncertain, infer a continent based on the text content."
            "Ensure the output contains only the three terms in the specified format, nothing else."
            )
        },
        {"role": "user", "content": f"Here is a txt to analyze: {final_summary}"},
    ]
    response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages = messages,
            temperature=0,
            max_tokens=200
        )
    reply = response["choices"][0]["message"]["content"]
    return image_prefix, final_summary, reply

def save_result(prefix, summary, summary_final, output_dir, output_dir1):
    """保存单张图片的结果到文件中"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{prefix}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
        
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)

    output_path = os.path.join(output_dir1, f"{prefix}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_final)

def main():
    # 图片文件路径列表
    image_dir = "data/panoramas"  # 确保这个路径是正确的
    image_paths = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) if img.endswith("_east.jpg")]

    output_dir = "data/tests"  # 修改保存路径到 data/GPT
    output_dir1 = "data/test"

    for image_path in image_paths:
        prefix = os.path.basename(image_path).replace("_east.jpg", "")
        output_path = os.path.join(output_dir, f"{prefix}.txt")

        # 如果文件已存在，则跳过处理
        if os.path.exists(output_path):
            print(f"已跳过图片 {image_path}，结果文件已存在。")
            continue

        prefix, summary, summary_final = process_image(image_path)  # 处理单张图片
        save_result(prefix, summary, summary_final, output_dir, output_dir1)  # 保存结果
        print(f"已处理图片 {image_path} 并保存结果到 {output_dir}。")

if __name__ == "__main__":
    main()
