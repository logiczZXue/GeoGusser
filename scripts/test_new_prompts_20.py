"""Test new JSON prompts on 20 diverse images with FULL sensor fusion."""
import sys, json, time, re, math, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn
from regression.element_fusion import fuse_elements_v3, extract_all_elements, parse_geocot_json, score_sensor_consistency
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate

MODEL_PATH = "D:/Geocomp/models/Qwen3-VL-4B-Instruct/qwen/Qwen3-VL-4B-Instruct"
PROMPTS_DIR = str(Path(__file__).resolve().parent.parent / "src" / "Geocot" / "prompts")
N_IMAGES = 20

random.seed(42)


def haversine(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2): return float("inf")
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_diverse_images():
    with open("D:/Geocomp/data/merged_geotagged.json", encoding="utf-8") as f:
        gt = json.load(f)
    candidates = [(p, c[0], c[1]) for p, c in gt.items() if Path(p).exists()]
    random.shuffle(candidates)

    bands = {">40N": [], "35-40N": [], "30-35N": [], "25-30N": [], "<25N": []}
    for p, lat, lng in candidates:
        if lat > 40: bands[">40N"].append((p, lat, lng))
        elif lat > 35: bands["35-40N"].append((p, lat, lng))
        elif lat > 30: bands["30-35N"].append((p, lat, lng))
        elif lat > 25: bands["25-30N"].append((p, lat, lng))
        else: bands["<25N"].append((p, lat, lng))

    picks = []
    for band, items in bands.items():
        picks.extend(items[:4])
    return picks[:N_IMAGES]


def run_one_inference(img, pipeline, sensor_elev, sensor_temp, sensor_humid):
    """Single VLM inference with JSON retry. Returns (elements, dt, n_retries, all_ok)."""
    t0 = time.time()
    n_retries = 0
    max_retries = 1

    while True:
        result = pipeline.run(
            img,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
        )
        mj = parse_geocot_json(result.stage_outputs.get("macro", ""))
        rj = parse_geocot_json(result.stage_outputs.get("regional", ""))
        lj = parse_geocot_json(result.stage_outputs.get("local", ""))

        if all([mj, rj, lj]) or n_retries >= max_retries:
            break
        n_retries += 1
        print(f"    JSON parse failed, retry {n_retries}/{max_retries}...", flush=True)

    elements = extract_all_elements(mj or {}, rj or {}, lj or {})
    dt = time.time() - t0
    return elements, dt, n_retries, all([mj, rj, lj])


def main():
    images = load_diverse_images()
    print(f"Test images: {len(images)}")
    for i, (p, lat, lng) in enumerate(images):
        print(f"  {i+1}. {Path(p).name} ({lat:.2f}, {lng:.2f})")

    # Load models
    print("\nLoading DEM + Climate...")
    dem = get_dem()
    climate = get_climate()
    print(f"  DEM: {'OK' if dem else 'MISSING'}, Climate: {'OK' if climate else 'MISSING'}")

    print("\nLoading VLM...")
    model, processor, _ = load_qwen2vl(
        model_name=MODEL_PATH, load_in_4bit=True,
        max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    model_fn = create_qwen2vl_model_fn(model, processor, max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": PROMPTS_DIR})

    results = []
    total_time = 0
    month = 7  # July for temperature/humidity estimation

    for i, (img_path, true_lat, true_lng) in enumerate(images):
        name = Path(img_path).name
        print(f"\n[{i+1:2d}/{N_IMAGES}] {name} ({true_lat:.2f}, {true_lng:.2f})", flush=True)

        # ── Sensor simulation from GPS ──────────────────────────────────────
        sensor_elev = dem.query(true_lat, true_lng) if dem else None
        sensor_temp = None
        sensor_humid = None
        if climate:
            elev_for_climate = sensor_elev if sensor_elev else 500
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, elev_for_climate, month)
            sensor_temp = (t_lo + t_hi) / 2
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            sensor_humid = (h_lo + h_hi) / 2

        sensor_str = f"elev={sensor_elev:.0f}m" if sensor_elev else "elev=N/A"
        if sensor_temp: sensor_str += f" T={sensor_temp:.0f}°C"
        if sensor_humid: sensor_str += f" RH={sensor_humid:.0f}%"
        print(f"  Sensors: {sensor_str}", flush=True)

        # ── Adaptive N=2: sensor-scored VLM inference ────────────────────────
        img = Image.open(img_path).convert("RGB")

        # First inference
        elements, dt, n_retries, all_ok = run_one_inference(
            img, pipeline, sensor_elev, sensor_temp, sensor_humid)
        sensor_score = score_sensor_consistency(
            elements, sensor_elev, sensor_temp, sensor_humid)
        n_samples = 1

        # Adaptive second inference if sensor score is low
        if sensor_score < 0.80:
            print(f"  Score={sensor_score:.2f}<0.80 → 2nd inference...", flush=True)
            elements2, dt2, n_retries2, all_ok2 = run_one_inference(
                img, pipeline, sensor_elev, sensor_temp, sensor_humid)
            sensor_score2 = score_sensor_consistency(
                elements2, sensor_elev, sensor_temp, sensor_humid)
            dt += dt2
            n_retries += n_retries2
            n_samples = 2
            if sensor_score2 > sensor_score:
                elements = elements2
                all_ok = all_ok2
                sensor_score = sensor_score2

        total_time += dt
        vlm_elev = elements.get("elevation_estimate_m", "N/A")

        json_status = "OK" if all_ok else "MISSING"
        if n_retries > 0:
            json_status += f" (retry={n_retries})"
        if n_samples > 1:
            json_status += f" N={n_samples}"

        # ── Fusion with sensor data ────────────────────────────────────────
        fusion = fuse_elements_v3(
            elements,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
        )
        pred_lat = fusion.latitude
        pred_lng = fusion.longitude
        err = haversine(true_lat, true_lng, pred_lat, pred_lng) if pred_lat and pred_lng else 9999

        search_area_str = "N/A"
        for line in fusion.explanation_parts:
            if line.startswith("[SEARCH-AREA]"):
                search_area_str = line.split("Union:")[1].strip() if "Union:" in line else line
                break

        print(f"  {dt:.0f}s  VLM_elev={vlm_elev}  score={sensor_score:.2f}  JSON:{json_status}")
        print(f"  Fusion:({pred_lat:.2f},{pred_lng:.2f}) err={err:.0f}km  {search_area_str}", flush=True)

        results.append({
            "image": name,
            "true_lat": true_lat, "true_lng": true_lng,
            "sensor_elevation_m": sensor_elev,
            "sensor_temperature_c": sensor_temp,
            "sensor_humidity_pct": sensor_humid,
            "vlm_elevation_estimate": vlm_elev,
            "pred_lat": pred_lat, "pred_lng": pred_lng,
            "error_km": err,
            "uncertainty_km": fusion.uncertainty_km,
            "confidence": fusion.confidence,
            "time_s": dt,
            "json_ok": all_ok,
            "n_retries": n_retries,
            "n_samples": n_samples,
            "sensor_score": sensor_score,
            "n_elements": len(elements),
            "explanation": fusion.explanation_parts,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    errors = [r["error_km"] for r in results if r["error_km"] < 9999]
    errors.sort()
    print(f"Total time: {total_time:.0f}s ({total_time/N_IMAGES:.0f}s/image)")
    print(f"JSON valid: {sum(1 for r in results if r['json_ok'])}/{N_IMAGES}")
    print(f"Errors (km):")
    print(f"  Mean:  {sum(errors)/len(errors):.0f}")
    print(f"  Median:{errors[len(errors)//2]:.0f}")
    print(f"  Best:  {min(errors):.0f}")
    print(f"  Worst: {max(errors):.0f}")
    print(f"  <=20km: {sum(1 for e in errors if e<=20)}/{len(errors)}")
    print(f"  <=100km:{sum(1 for e in errors if e<=100)}/{len(errors)}")
    print(f"  <=200km:{sum(1 for e in errors if e<=200)}/{len(errors)}")

    out = "D:/Geocomp/output/new_prompt_20_test_sensors.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
