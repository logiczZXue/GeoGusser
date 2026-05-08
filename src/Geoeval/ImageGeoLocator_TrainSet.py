import openai
from PIL import Image
import base64
import os
from io import BytesIO

# 设置 OpenAI API 密钥
openai.api_key = ""

def resize_image(image_path, size=(256, 256)):
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

    questions =  [
        """Locate the continent, country and city, and summarize it into a paragraph (note: it must be a paragraph).  Let's think step by step"""    
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
                "Extract the city, country, and continent from the result. "
                "Output them strictly in the order: city, country, continent. "
                "Separate them with a comma and a space. "
                "If any of these is not found, write 'none' in its place. "
                "Ensure the output contains only the three terms in the specified format, nothing else."
            )
        },
        {"role": "user", "content": f"Here is a txt to analyze: {final_summary}"},
    ]
    response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages = messages,
            temperature=0,
            max_tokens=500
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

    output_dir = "data/zeros"  # 修改保存路径到 data/GPT
    output_dir1 = "data/zero"

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
