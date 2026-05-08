import openai
import os

# 设置 OpenAI API 密钥
openai.api_key = ""

def process_image(image_path):
    """处理单张图片并与 GPT 进行一轮对话"""
    # 提取图片文件的前半部分命名（不包括后缀和 east 等后缀）
    image_prefix = os.path.basename(image_path).replace("_east.jpg", "")

    # 提供图像路径或描述（避免直接传递 Base64 编码）
    image_description = f"The image is stored at: {image_path}"

    # 初始化对话消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant for analyzing images."},
        {"role": "user", "content": image_description},
        {"role": "user", "content": (
            "Locate the location of the picture as accurately as possible. "
            "Locate the continent, country and city, and summarize it into a paragraph. "
            "Let's think step by step."
        )},
    ]

    # 提问并获取回答
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
        max_tokens=500
    )
    reply = response["choices"][0]["message"]["content"]

    return image_prefix, reply

def save_result(prefix, summary, output_dir):
    """保存单张图片的结果到文件中"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{prefix}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

def main():
    # 图片文件路径列表
    image_dir = "data/panoramas"  # 确保这个路径是正确的
    output_dir = "data/zero"  # 修改保存路径到 data/zero
    image_paths = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) if img.endswith("_east.jpg")]

    for image_path in image_paths:
        # 检查输出目录中是否已存在对应的 .txt 文件
        image_prefix = os.path.basename(image_path).replace("_east.jpg", "")
        output_path = os.path.join(output_dir, f"{image_prefix}.txt")

        if os.path.exists(output_path):
            print(f"图片 {image_path} 已处理过，跳过。")
            continue  # 跳过已经处理过的图片

        # 处理图片并保存结果
        prefix, summary = process_image(image_path)
        save_result(prefix, summary, output_dir)
        print(f"已处理图片 {image_path} 并保存结果到 {output_dir}。")

if __name__ == "__main__":
    main()
