"""Quick test: download Qwen2-VL-2B and run 3-stage GeoCoT inference.

Uses the 2B model instead of 7B because RTX 4060 Laptop has 8GB VRAM.
The 2B model at bfloat16 fits comfortably (~4GB), leaving room for KV cache.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
import time

# Use 2B model for 8GB VRAM compatibility
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

print("=== Step 1: Loading model ===")
print(f"Model: {MODEL_NAME}")
t0 = time.time()

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# 2B model fits in bfloat16 on 8GB VRAM (~4GB)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
t1 = time.time()
print(f"Model loaded in {t1-t0:.0f}s")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n=== Step 2: Preparing test image ===")
test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Paris_street_view%2C_France.jpg/1280px-Paris_street_view%2C_France.jpg"
try:
    image = Image.open(requests.get(test_url, stream=True, timeout=15).raw).convert("RGB")
    print("Downloaded test street view image (Paris)")
except Exception:
    image = Image.new("RGB", (640, 480), (128, 128, 128))
    print("Using blank test image (network unavailable)")

from qwen_vl_utils import process_vision_info

def run_stage(image, prompt_text, stage_name, max_tokens=200):
    """Run a single GeoCoT stage."""
    print(f"\n--- {stage_name} ---")
    t_start = time.time()

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    result = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]

    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.1f}s | Tokens generated: ~{len(result.split())}")
    print(f"  Result: {result[:300]}...")
    return result

# Stage 1: Macro
macro_result = run_stage(image, (
    "Analyze this street view image for broad geographic context. Consider:\n"
    "- Climate zone (tropical, temperate, arid, polar) based on vegetation, sky, and soil\n"
    "- General topography (flat, mountainous, coastal, inland)\n"
    "- Vegetation type (palm trees, conifers, deciduous, sparse desert)\n"
    "- Overall urbanization level (dense city, suburban, rural)\n\n"
    "Provide a brief assessment of which continent(s) and broad region(s) this image could be from."
), "Stage 1: Macro (Continent-level)", max_tokens=200)

# Stage 2: Regional
regional_result = run_stage(image, (
    f"Based on the macro-level assessment:\n\"{macro_result}\"\n\n"
    "Now examine this image for country-specific indicators:\n"
    "- Language on signs, buildings, or vehicles\n"
    "- Architectural style (roof shapes, wall colors, building materials)\n"
    "- Traffic direction and road infrastructure\n"
    "- License plate format and color\n"
    "- Utility pole design and fire hydrant style\n\n"
    "Which specific country does this image most likely come from?"
), "Stage 2: Regional (Country-level)", max_tokens=200)

# Stage 3: Local
local_result = run_stage(image, (
    f"Based on the analysis so far:\n- Macro: {macro_result}\n- Regional: {regional_result}\n\n"
    "Now identify city-level details and provide your final geolocation.\n"
    "Format: This image was most likely taken in [city], [country], [continent]."
), "Stage 3: Local (City-level)", max_tokens=300)

print("\n" + "=" * 60)
print("GeoCoT 3-Stage Pipeline Test COMPLETE!")
print("=" * 60)
print(f"\nStage 1 (Macro):    {macro_result[:200]}...")
print(f"\nStage 2 (Regional): {regional_result[:200]}...")
print(f"\nStage 3 (Local):    {local_result}")
print(f"\nFinal prediction:   {local_result}")
print(f"\nTotal GPU memory:   {torch.cuda.memory_allocated() / 1e9:.2f} GB")
