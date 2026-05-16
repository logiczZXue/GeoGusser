"""Evaluate LoRA vs base VLM on GeoCoT output quality."""
import os, sys, time, math
from pathlib import Path
import torch
from PIL import Image

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from Geocot.Geocot import GeoCoTPipeline, create_qwen2vl_model_fn


def load_lora_model(base_model_name="Qwen/Qwen2-VL-2B-Instruct",
                    adapter_path="output/vlm_lora/adapter"):
    """Load base model + LoRA adapter."""
    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        min_pixels=256*28*28,
        max_pixels=12845056,
        local_files_only=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    model_fn = create_qwen2vl_model_fn(model, processor)
    return model, processor, model_fn


def main():
    # Test image — Huangshan karst
    img_path = "data/mapillary/1163967815472102.jpg"
    true_lat, true_lng = 30.0961, 118.1802

    if not os.path.exists(img_path):
        # Fallback: any existing image
        import json
        with open("data/merged_geotagged.json") as f:
            data = json.load(f)
        for p in data:
            if os.path.exists(p):
                img_path = p
                true_lat, true_lng = data[p][0], data[p][1]
                break

    print(f"Image: {img_path}")
    print(f"Truth: ({true_lat:.4f}, {true_lng:.4f})")
    image = Image.open(img_path).convert("RGB")

    # ── Base model first ────────────────────────────────────────────
    print("\n" + "="*70)
    print("Base Model GeoCoT Output")
    print("="*70)
    from Geocot.Geocot import load_qwen2vl
    base_model, base_proc, base_fn = load_qwen2vl("Qwen/Qwen2-VL-2B-Instruct")
    p_base = GeoCoTPipeline(base_fn)
    t0 = time.time()
    result_base = p_base.run(image)
    t_base = time.time() - t0

    for stage in ["macro", "regional", "local"]:
        out = result_base.stage_outputs.get(stage, "")
        print(f"\n--- {stage.upper()} ({len(out)} chars) ---")
        print(out[:500])

    # Free base model memory
    del base_model, base_proc, base_fn, p_base
    torch.cuda.empty_cache()

    # ── LoRA model ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LoRA GeoCoT Output")
    print("="*70)
    _, _, lora_fn = load_lora_model()
    p_lora = GeoCoTPipeline(lora_fn)
    t0 = time.time()
    result_lora = p_lora.run(image)
    t_lora = time.time() - t0

    for stage in ["macro", "regional", "local"]:
        out = result_lora.stage_outputs.get(stage, "")
        print(f"\n--- {stage.upper()} ({len(out)} chars) ---")
        print(out[:500])

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"LoRA time: {t_lora:.1f}s | Base time: {t_base:.1f}s")

    l_lat = result_lora.final_prediction.latitude
    l_lng = result_lora.final_prediction.longitude
    b_lat = result_base.final_prediction.latitude
    b_lng = result_base.final_prediction.longitude

    def haversine(lat1, lng1, lat2, lng2):
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) * math.sin(dlng/2)**2)
        return 6371.0 * 2 * math.asin(math.sqrt(a))

    if l_lat and l_lng:
        print(f"LoRA coords: ({l_lat:.4f}, {l_lng:.4f}) error={haversine(l_lat, l_lng, true_lat, true_lng):.1f}km")
        print(f"LoRA location: {result_lora.final_prediction.city}, {result_lora.final_prediction.country}")
    if b_lat and b_lng:
        print(f"Base coords: ({b_lat:.4f}, {b_lng:.4f}) error={haversine(b_lat, b_lng, true_lat, true_lng):.1f}km")
        print(f"Base location: {result_base.final_prediction.city}, {result_base.final_prediction.country}")


if __name__ == "__main__":
    main()
