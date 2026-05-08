import os
import sys

import torch
import pandas as pd
import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder
from peft import PeftModel
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import dotenv
from openai import OpenAI

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
dotenv.load_dotenv("../../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)
temperature = 0.7
top_p = 0.9
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
csv_path = "/home/xiuying.chen/jingpu/data/im2gps3k.csv"
# save_dir = "/home/xiuying.chen/jingpu/data/llama"
save_dir2 = "/home/xiuying.chen/jingpu/data/llamas_im2gps3k"
# prompt = """
# Here are five pictures, labeled south, east, north, west, and panorama. The five images were taken in the same place. Based on these images, identify the city, country, and continent. Please analyze the geographic elements in these images—such as landmarks, architecture, language, or other visual clues—that can help you determine the location. 
# Your answer must follow this exact format: "city, country, continent" in a single line.
# You should answer in English.
# If you cannot determine the location, please guess the answer from continent to city.
# Do not include any extra symbols, text, or explanations even if you cannot determine the location.
# """
# prompt = """
# Analyze the following five images labeled south, east, north, west, and panorama, all taken from the same location. Use geographic elements such as landmarks, architecture, language, or other visual clues to determine the location.
# If you cannot determine the location, please guess the answer from continent to city. Do not answer you cannot determine the location.
# Output the chain of reasoning.
# """
prompt = """
Analyze the following image. Use geographic elements such as landmarks, architecture, language, or other visual clues to determine the location.
If you cannot determine the location, please guess the answer from continent to city. Do not answer you cannot determine the location.
Output the chain of reasoning.
"""

def ask_gpt(response):
    prompt = f"""Analyze the txt: {response}. Filter out the city, country, and continent from the txt. Answer in the format: 'city, country, continent'.
    If there is no name, fill in 'Unknown'.
    Attention, your answer should be only one line (while input may be multiple lines) format 'city, country, continent'.
    Example:'New York, United States, North America'"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant at geograhic location."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)

def load_model_and_processor(model_name: str, finetuning_path: str = None):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        token=hf_token,
    )
    processor = MllamaProcessor.from_pretrained(
        model_name, token=hf_token, use_safetensors=True
    )

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model, finetuning_path, is_adapter=True, torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")

    model, processor = accelerator.prepare(model, processor)
    return model, processor

def process_image(image_path: str = None, image=None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")

def generate_text_from_image(
    model, processor, images, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from image using model"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"} for _ in range(1)  # 生成五个图像类型的字典
                ] + [{"type": "text", "text": prompt_text}]
            ,
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    
    # 将图像放入一个列表中
    inputs = processor(
        images, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
    ).to(device)  # 这里将 images 作为一个列表传递
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS
    )
    return processor.decode(output[0])[len(prompt) :]

def get_location(panoid, model, processor):
    images = []
    # for direction in ["north", "east", "south", "west", "panoramic"]:
    #     image_path = f"/home/xiuying.chen/jingpu/data/panoramas/{panoid}_{direction}.jpg"
    #     image = process_image(image_path=image_path)
    #     images.append(image)
    image = process_image(image_path=f"/home/xiuying.chen/jingpu/data/im2gps3k/{panoid}.jpg")
    images.append(image)
    result = ""
    # for index in range(5):
    #     result = generate_text_from_image(
    #         model, processor, images, prompt, temperature, top_p
    #     ).replace("<|eot_id|>", "")
    #     if result.endswith("."):
    #         result = result.replace(".", "")
        
    #     parts = result.split(",")
    #     if len(parts) == 3 and parts[2].strip() in ["North America", "South America", "Europe", "Africa", "Asia", "Oceania"]:
    #         break
    result = generate_text_from_image(
        model, processor, images, prompt, temperature, top_p
    ).replace("<|eot_id|>", "")
    if result.endswith("."):
        result = result.replace(".", "")
    with open(f"{save_dir2}/{panoid}.txt", 'w', encoding='utf-8') as f:
        f.write(result)

    # result = ask_gpt(result)
    # with open(f"{save_dir}/{panoid}.txt", 'w', encoding='utf-8') as f:
    #     f.write(result) 

if __name__ == "__main__":
    model, processor = load_model_and_processor(model_name)
    data = pd.read_csv(csv_path)
    for panoid in tqdm.tqdm(data["panoID"],total=len(data["panoID"]), desc="Processing panoramas"):
        get_location(panoid, model, processor)