"""Compare old vs enhanced GeoCoT prompts on a single image."""
import json, math, os, sys, time
from pathlib import Path
import torch
from PIL import Image

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from Geocot.Geocot import (
    GeoCoTPipeline, GeoCoTPrompt, GeoCoTStage, GeoCoTResult,
    create_qwen2vl_model_fn, load_qwen2vl,
)

def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def main():
    # Image: karst near Huangshan
    img_path = "data/mapillary/1163967815472102.jpg"
    true_lat, true_lng = 30.0961, 118.1802

    if not os.path.exists(img_path):
        # Try other karst images
        with open("output/regression/scene_labels.json") as f:
            labels = json.load(f)
        for p, st in labels.items():
            if int(st) == 1 and os.path.exists(p):
                img_path = p
                with open("data/merged_geotagged.json") as f:
                    data = json.load(f)
                true_lat, true_lng = data[p][0], data[p][1]
                break

    print(f"Image: {img_path}")
    print(f"True: ({true_lat:.4f}, {true_lng:.4f})")

    # Load VLM
    print("Loading VLM...")
    model, processor, model_fn = load_qwen2vl()  # default: Qwen3-VL-2B
    image = Image.open(img_path).convert("RGB")

    # Test enhanced prompts
    print("\n" + "=" * 60)
    print("ENHANCED PROMPTS")
    print("=" * 60)
    p_enhanced = GeoCoTPipeline(model_fn)
    t0 = time.time()
    result_e = p_enhanced.run(image)
    t_e = time.time() - t0

    for stage in GeoCoTStage:
        out = result_e.stage_outputs.get(stage.value, "")
        # Print first 300 chars
        print(f"\n[{stage.value.upper()}] ({len(out)} chars)")
        print(out[:400])

    print(f"\nEnhanced time: {t_e:.1f}s")
    print(f"Enhanced scene_type: {result_e.scene_type}")
    print(f"Enhanced LOCATION: {result_e.final_prediction}")

    # Test with disabled knowledge injection (old-style prompts)
    print("\n" + "=" * 60)
    print("OLD PROMPTS (knowledge disabled)")
    print("=" * 60)
    p_old = GeoCoTPipeline(model_fn, {"enable_knowledge": False, "prompts_dir": "/nonexistent"})
    t0 = time.time()
    result_o = p_old.run(image)
    t_o = time.time() - t0

    for stage in GeoCoTStage:
        out = result_o.stage_outputs.get(stage.value, "")
        print(f"\n[{stage.value.upper()}] ({len(out)} chars)")
        print(out[:400])

    print(f"\nOld time: {t_o:.1f}s")
    print(f"Old scene_type: {result_o.scene_type}")
    print(f"Old LOCATION: {result_o.final_prediction}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Enhanced time: {t_e:.1f}s | Old time: {t_o:.1f}s")

    e_lat = result_e.final_prediction.latitude
    e_lng = result_e.final_prediction.longitude
    o_lat = result_o.final_prediction.latitude
    o_lng = result_o.final_prediction.longitude

    if e_lat and e_lng:
        e_err = haversine_km(e_lat, e_lng, true_lat, true_lng)
        print(f"Enhanced coords: ({e_lat:.4f}, {e_lng:.4f}) error={e_err:.1f}km")
    if o_lat and o_lng:
        o_err = haversine_km(o_lat, o_lng, true_lat, true_lng)
        print(f"Old coords:      ({o_lat:.4f}, {o_lng:.4f}) error={o_err:.1f}km")

if __name__ == "__main__":
    main()
