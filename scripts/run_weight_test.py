"""Quick test of per-element sensor-consistency weighting on 3 images."""
import sys, os, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from regression.element_fusion import run_fusion_pipeline_v3
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate
from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn

model_name = 'D:/Geocomp/models/Qwen3-VL-4B-Instruct/qwen/Qwen3-VL-4B-Instruct'
model, processor, _ = load_qwen2vl(model_name=model_name, load_in_4bit=True,
                                    max_new_tokens=1024, temperature=0.7, top_p=0.8,
                                    enable_thinking=False)
model_fn = create_qwen2vl_model_fn(model, processor,
                                    max_new_tokens=1024, temperature=0.7, top_p=0.8,
                                    enable_thinking=False)
pipeline = GeoCoTPipeline(model_fn, {'prompts_dir': os.path.join('src', 'Geocot', 'prompts')})
dem = get_dem()
climate = get_climate()

test_cases = [
    ('D:/Geocomp/data/mapillary/461163799153219.jpg', (30.5554, 103.9990)),   # Chengdu
    ('D:/Geocomp/data/mapillary/3002140840019379.jpg', (31.1595, 121.3036)),  # Shanghai
    ('D:/Geocomp/data/mapillary/149407033803389.jpg', (39.8295, 116.2878)),   # Beijing
]

results = []
for i, (path, (true_lat, true_lng)) in enumerate(test_cases):
    name = os.path.basename(path)
    sensor_elev = dem.query(true_lat, true_lng)
    month = 7
    t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, sensor_elev, month)
    sensor_temp = round((t_lo + t_hi) / 2, 1)
    h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
    sensor_humid = round((h_lo + h_hi) / 2, 1)

    print(f"\n{'='*70}")
    print(f"[{i+1}/3] {name}")
    print(f"  True: ({true_lat:.4f}, {true_lng:.4f}) | DEM: {sensor_elev:.0f}m | {sensor_temp:.0f}C | {sensor_humid:.0f}%RH")

    image = Image.open(path).convert('RGB')
    t0 = time.time()
    geocot_result = pipeline.run(image,
                                  sensor_elevation_m=sensor_elev,
                                  sensor_temperature_c=sensor_temp,
                                  sensor_humidity_pct=sensor_humid)
    dt = time.time() - t0

    geo_pred = geocot_result.final_prediction
    geo_raw = (geo_pred.latitude, geo_pred.longitude) if geo_pred.latitude is not None else None
    if geo_raw and geo_raw[0] is not None:
        geocot_raw_err = 111.32 * math.sqrt(
            (geo_raw[0]-true_lat)**2 +
            (geo_raw[1]-true_lng)**2 * math.cos(math.radians(true_lat))**2
        )
        print(f"  GeoCoT raw : ({geo_raw[0]:.2f}, {geo_raw[1]:.2f}) -> error {geocot_raw_err:.0f} km")

    fusion_result = run_fusion_pipeline_v3(
        macro_text=geocot_result.stage_outputs.get('macro', ''),
        regional_text=geocot_result.stage_outputs.get('regional', ''),
        local_text=geocot_result.stage_outputs.get('local', ''),
        sensor_elevation_m=sensor_elev,
        sensor_temperature_c=sensor_temp,
        sensor_humidity_pct=sensor_humid,
        geocot_prediction=geo_raw,
    )

    result = fusion_result
    fusion_err = 111.32 * math.sqrt(
        (result.latitude-true_lat)**2 +
        (result.longitude-true_lng)**2 * math.cos(math.radians(true_lat))**2
    )
    print(f"  Fusion     : ({result.latitude:.2f}, {result.longitude:.2f}) +/-{result.uncertainty_km:.0f}km -> error {fusion_err:.0f} km")
    print(f"  Confidence : {result.confidence:.1%} | {dt:.1f}s")
    results.append({
        'image': name, 'true': (true_lat, true_lng),
        'geocot_raw': geo_raw, 'geocot_err': geocot_raw_err if geo_raw else None,
        'fusion': (result.latitude, result.longitude), 'fusion_err': fusion_err,
        'confidence': result.confidence, 'time': dt,
    })

if results:
    errors = [r['fusion_err'] for r in results]
    print(f"\n{'='*70}")
    print(f"Summary (n={len(results)}):")
    print(f"  Mean error: {sum(errors)/len(errors):.0f} km")
    print(f"  Median: {sorted(errors)[len(errors)//2]:.0f} km")
