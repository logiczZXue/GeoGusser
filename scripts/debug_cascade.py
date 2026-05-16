"""Debug: inspect GeoCoT output and cascade constraint placement for a single image."""
import json, math, os, sys, time
from pathlib import Path
import torch
from PIL import Image

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.pipeline import load_geocot_with_regression


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def main():
    # Find a karst test image
    with open("output/regression/scene_labels.json") as f:
        labels = json.load(f)
    with open("data/merged_geotagged.json") as f:
        data = json.load(f)

    karst_images = []
    for path, st in labels.items():
        if int(st) == 1 and path in data and os.path.exists(path):
            karst_images.append((path, data[path][0], data[path][1]))
    if not karst_images:
        print("No karst images found")
        return

    img_path, true_lat, true_lng = karst_images[0]
    print(f"Test image: {img_path}")
    print(f"True coords: ({true_lat:.4f}, {true_lng:.4f})")

    # Load models
    print("\nLoading VLM + MoE...")
    model, processor, pipeline = load_geocot_with_regression(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        moe_path="output/moe/moe_final.pt",
        load_in_4bit=False,
        enable_knowledge=True,
    )

    image = Image.open(img_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Run cascade
    print("\n=== CASCADE RUN ===")
    torch.cuda.empty_cache()
    result = pipeline.run(image, use_cascade=True, use_moe=False)

    # Inspect GeoCoT output
    print("\n--- GeoCoT MACRO ---")
    print(result.stage_outputs.get("macro", "N/A")[:500])
    print("\n--- GeoCoT REGIONAL ---")
    print(result.stage_outputs.get("regional", "N/A")[:500])
    print("\n--- GeoCoT LOCAL ---")
    print(result.stage_outputs.get("local", "N/A")[:500])

    # Parse entities from each stage
    from Geocot.structured_parser import parse_geocot_output
    stage_outputs = {
        k: v for k, v in result.stage_outputs.items()
        if k in ("macro", "regional", "local")
    }

    print("\n--- PARSED ENTITIES (all stages) ---")
    entities = parse_geocot_output(stage_outputs)
    print(f"  Terrain: {entities.terrain_types}")
    print(f"  Climate: {entities.climate_zones}")
    print(f"  Vegetation: {entities.vegetation_types}")
    print(f"  Architecture: {entities.architectural_styles}")
    print(f"  Cultural: {entities.cultural_markers}")
    print(f"  Named locations: {entities.named_locations}")
    print(f"  Scene hint: {entities.scene_type_hint}")

    # Check KB matches
    from Geocot.spatial_kb import entities_to_spatial_regions
    regions = entities_to_spatial_regions(entities, top_k=5)
    print(f"\n--- KB MATCHED REGIONS ({len(regions)}) ---")
    for r in regions:
        err = haversine_km(r.center_lat, r.center_lng, true_lat, true_lng)
        print(f"  {r.source} ({r.source_type}): ({r.center_lat:.3f}, {r.center_lng:.3f}) r={r.radius_km:.0f}km conf={r.confidence:.2f} err_from_truth={err:.1f}km")

    # Cascade constraint
    cascade_info = result.stage_outputs.get("cascade", "")
    print(f"\n--- CASCADE INFO ---")
    print(cascade_info)

    # Final prediction
    pred = result.final_prediction
    err = haversine_km(pred.latitude, pred.longitude, true_lat, true_lng)
    print(f"\n--- FINAL ---")
    print(f"  Pred: ({pred.latitude:.4f}, {pred.longitude:.4f})")
    print(f"  True: ({true_lat:.4f}, {true_lng:.4f})")
    print(f"  Error: {err:.1f} km")

    # Also run MoE-only for comparison
    print("\n=== MOE-ONLY (for comparison) ===")
    features = pipeline._extractor.extract(image).float()
    lat, lng, weights, e_lats, e_lngs = pipeline._moe.predict(
        features.unsqueeze(0), return_expert_info=True
    )
    moe_err = haversine_km(lat.item(), lng.item(), true_lat, true_lng)
    print(f"  MoE pred: ({lat.item():.4f}, {lng.item():.4f}) error={moe_err:.1f}km")
    print(f"  Router weights: {weights.squeeze().tolist()}")
    print(f"  Top expert: {weights.argmax().item()} ({weights.max().item():.1%})")


if __name__ == "__main__":
    main()
