"""Test GeoCoT with a real street view image (standalone, no Geocot.py dependency).

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
MAX_IMAGE_PIXELS = 504 * 28 * 28  # ~629x629, fits in 8GB VRAM

# Updated parameters (2026-05-10): lower temp for more deterministic output
TEMPERATURE = 0.3
TOP_P = 0.85


def load_model():
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_IMAGE_PIXELS,
        local_files_only=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    print(f"Loaded in {time.time()-t0:.0f}s | GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, processor


def resize_image(image: Image.Image, max_pixels: int = 500000) -> Image.Image:
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized: {w}x{h} -> {new_w}x{new_h}")
    else:
        print(f"  Image size: {w}x{h} (no resize needed)")
    return image


def run_stage(model, processor, image, prompt, stage_name, max_tokens=250):
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

    torch.cuda.empty_cache()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )

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

    image = resize_image(image)

    # Stage 1: Macro (improved prompt)
    macro_prompt = (
        "You are an expert geolocation analyst. Examine this street view image for broad geographic context.\n\n"
        "Analyze ONLY what you can actually see. Report on:\n"
        "- Climate zone (vegetation type, sky/clouds, soil color)\n"
        "- Topography (flat, hilly, mountainous, coastal, inland)\n"
        "- Vegetation density and type (forest, scrub, grassland, farmland)\n"
        "- Urbanization level (rural, suburban, urban, megacity)\n"
        "- Road surface and condition\n"
        "- Apparent season\n\n"
        "Based on ALL observations, narrow to the most likely continent(s) or large region(s). "
        "Rank from most to least likely. Do NOT give a vague answer."
    )

    macro = run_stage(model, processor, image, macro_prompt, "Stage 1: MACRO (Continent)", 256)

    # Stage 2: Regional (improved prompt)
    regional_prompt = (
        f'Based on the macro-level assessment:\n"{macro}"\n\n'
        "Now examine this image for country-level indicators. For EACH category, report what you ACTUALLY see:\n\n"
        "1. LANGUAGE & TEXT: Script/alphabet on signs, buildings, vehicles\n"
        "2. ARCHITECTURE: Roof style, wall materials/colors, building height pattern, distinctive features\n"
        "3. ROADS & VEHICLES: Driving side, road marking style, vehicle makes, license plate format\n"
        "4. INFRASTRUCTURE: Utility pole design, street lights, fire hydrant style\n"
        "5. PEOPLE & CLOTHING (if visible)\n\n"
        "Identify the most likely COUNTRY (max 2 candidates). List specific clues for each.\n"
        "CRITICAL: Do NOT hallucinate clues. If uncertain, narrow to a specific sub-region."
    )

    regional = run_stage(model, processor, image, regional_prompt, "Stage 2: REGIONAL (Country)", 256)

    # Stage 3: Local (improved prompt, requires LOCATION: format)
    local_prompt = (
        f"Based on the analysis so far:\n- Macro: {macro}\n- Regional: {regional}\n\n"
        "Now synthesize ALL evidence into a precise geolocation. Consider city-level details:\n"
        "1. Street furniture (benches, trash bins, bus stops)\n"
        "2. Sidewalk/pavement material, tile pattern, color\n"
        "3. Signage style (European vs American vs Asian traffic sign standards)\n"
        "4. Commercial signs, chain stores, local business types\n"
        "5. Any visible landmarks or distinctive buildings\n"
        "6. Urban density and street layout pattern\n\n"
        "Write a reasoning paragraph connecting observations to inferences.\n"
        "Then end with EXACTLY this line (nothing after it):\n"
        "LOCATION: [city], [country], [continent]\n\n"
        "Continent must be one of: Asia, Africa, Europe, North America, South America, Oceania.\n"
        "If uncertain about the city, give your best estimate. Do NOT write 'Unknown'."
    )

    local = run_stage(model, processor, image, local_prompt, "Stage 3: LOCAL (City)", 350)

    print(f"\n{'#'*60}")
    print(f"# FINAL RESULT")
    print(f"{'#'*60}")
    print(f"  Macro:    {macro[:200]}...")
    print(f"  Regional: {regional[:200]}...")
    print(f"  Local:    {local}")
    print(f"  GPU mem:  {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
