"""Test GeoCoT with a real street view image.

Usage:
    python scripts/test_real_image.py --image my_photo.jpg
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# Limit image resolution to fit in 8GB VRAM (model ~4.4GB, leaves ~3.6GB for inference)
MAX_IMAGE_PIXELS = 504 * 28 * 28  # = 395136, roughly 629x629


def load_model():
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loaded in {time.time()-t0:.0f}s | GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, processor


def resize_image(image: Image.Image, max_pixels: int = 500000) -> Image.Image:
    """Resize image if too large, to avoid OOM."""
    w, h = image.size
    pixels = w * h
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}x{h} -> {new_w}x{new_h}")
    else:
        print(f"  Image size: {w}x{h} (no resize needed)")
    return image


def run_stage(model, processor, image, prompt, stage_name, max_tokens=200):
    print(f"\n{'='*50}")
    print(f"  {stage_name}")
    print(f"{'='*50}")
    t0 = time.time()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    # Clear cache before generation to free fragmented memory
    torch.cuda.empty_cache()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    result = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s] {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    args = parser.parse_args()

    model, processor = load_model()

    # Get image
    if args.image and os.path.exists(args.image):
        image = Image.open(args.image).convert("RGB")
        print(f"Using image: {args.image}")
    else:
        import requests
        urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Paris_street_view%2C_France.jpg/800px-Paris_street_view%2C_France.jpg",
        ]
        image = None
        for url in urls:
            try:
                resp = requests.get(url, stream=True, timeout=20)
                if resp.status_code == 200:
                    image = Image.open(resp.raw).convert("RGB")
                    print(f"Downloaded test image")
                    break
            except Exception:
                continue
        if image is None:
            print("Could not download test image.")
            print("Usage: python test_real_image.py --image YOUR_IMAGE.jpg")
            image = Image.new("RGB", (640, 480), (128, 128, 128))

    # CRITICAL: resize large images to avoid OOM on 8GB VRAM
    image = resize_image(image)

    # Stage 1: Macro
    macro = run_stage(model, processor, image, (
        "Analyze this street view image for broad geographic context. Consider:\n"
        "- Climate zone (tropical, temperate, arid, polar) based on vegetation, sky, and soil\n"
        "- General topography (flat, mountainous, coastal, inland)\n"
        "- Vegetation type (palm trees, conifers, deciduous, sparse desert)\n"
        "- Overall urbanization level (dense city, suburban, rural)\n\n"
        "Provide a brief assessment of which continent(s) and broad region(s) this image could be from."
    ), "Stage 1: MACRO (Continent-level)", 200)

    # Stage 2: Regional
    regional = run_stage(model, processor, image, (
        f"Based on the macro-level assessment:\n\"{macro}\"\n\n"
        "Now examine this image for country-specific indicators:\n"
        "- Language on signs, buildings, or vehicles\n"
        "- Architectural style (roof shapes, wall colors, building materials)\n"
        "- Traffic direction and road infrastructure\n"
        "- License plate format and color\n"
        "- Utility pole design and fire hydrant style\n\n"
        "Which specific country does this image most likely come from?"
    ), "Stage 2: REGIONAL (Country-level)", 200)

    # Stage 3: Local
    local = run_stage(model, processor, image, (
        f"Based on the analysis so far:\n- Macro: {macro}\n- Regional: {regional}\n\n"
        "Now identify city-level details and provide your final geolocation.\n"
        "Format: This image was most likely taken in [city], [country], [continent]."
    ), "Stage 3: LOCAL (City-level)", 300)

    print(f"\n{'#'*60}")
    print(f"# FINAL RESULT")
    print(f"{'#'*60}")
    print(f"  Macro:    {macro[:200]}...")
    print(f"  Regional: {regional[:200]}...")
    print(f"  Local:    {local}")
    print(f"  GPU mem:  {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
