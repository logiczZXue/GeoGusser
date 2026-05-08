"""Run zero-shot baseline geolocation (no CoT reasoning).

Compares against GeoCoT by using the same VLM with a simple direct prompt.

Usage:
    python -m src.baseline.run_baseline --image-dir data/panoramas --output-dir output/baseline
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image
from src.utils.config import load_config

ZERO_SHOT_PROMPT = (
    "Analyze this street view image and identify the city, country, and continent. "
    "Consider geographic elements such as landmarks, architecture, language, and other visual clues. "
    "Your answer must follow this exact format: 'city, country, continent' in a single line. "
    "Do not include any extra text or explanations."
)


def run_baseline(model_fn, image_dir: str, output_dir: str):
    """Run zero-shot baseline on all images in a directory."""
    os.makedirs(os.path.join(output_dir, "reasoning"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_east" in f:
                images.append(os.path.join(root, f))

    if not images:
        # Fallback: any image
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(root, f))
                    break

    print(f"Running baseline on {len(images)} image(s)...")
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {name}...", end=" ", flush=True)

        image = Image.open(img_path).convert("RGB")
        result = model_fn(image, ZERO_SHOT_PROMPT, {})

        # Save reasoning (even though it's direct, we keep the format consistent)
        with open(os.path.join(output_dir, "reasoning", f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(result)

        # Extract and save prediction
        from src.Geocot.prediction_extractor import extract_prediction_enhanced, format_prediction
        pred = extract_prediction_enhanced(result)
        with open(os.path.join(output_dir, "predictions", f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(format_prediction(pred))

        print(f"-> {format_prediction(pred)}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run zero-shot baseline geolocation")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/baseline")
    parser.add_argument("--config", type=str, default="default")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from src.Geocot.Geocot import create_hf_model_fn
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from accelerate import Accelerator

    accelerator = Accelerator()
    device = accelerator.device
    model_name = cfg["model"]["name"]

    print(f"Loading model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_name)

    model_fn = create_hf_model_fn(model, processor, device)
    run_baseline(model_fn, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
