"""Quick test: Xishuangbanna temple image with updated macro prompt."""
import sys
from pathlib import Path
_scripts_dir = Path(__file__).resolve().parent
_src_dir = _scripts_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn, resize_image_for_vlm

img_path = "D:/Geocomp/data/mapillary/277864400647976.jpg"

print("Loading model...")
model, processor, _ = load_qwen2vl(
    model_name="D:/Geocomp/models/Qwen3-VL-4B-Thinking",
    load_in_4bit=True,
    max_new_tokens=4096,
    temperature=0.6,
    top_p=0.8,
    enable_thinking=True,
)
model_fn = create_qwen2vl_model_fn(model, processor, max_new_tokens=4096, temperature=0.6, top_p=0.8, enable_thinking=True)
pipeline = GeoCoTPipeline(model_fn, {'prompts_dir': str(_src_dir / 'Geocot' / 'prompts')})

image = Image.open(img_path).convert("RGB")
print(f"Image: {Path(img_path).name}")

# Sensor data: lat=21.991, lng=100.805 → estimate elevation
from regression.dem_lookup import get_dem
from regression.climate_lookup import ClimateValidator
dem = get_dem()
climate = ClimateValidator()
elev = dem.query(21.991, 100.805)
t_lo, t_hi = climate.estimate_temperature_range(21.991, 100.805, elev, 7)
h_lo, h_hi = climate.estimate_humidity_range(21.991, 100.805, 7)
sensor_temp = round((t_lo + t_hi) / 2, 1)
sensor_humid = round((h_lo + h_hi) / 2, 1)
print(f"Sensor: elev={elev:.0f}m, temp={sensor_temp}C, humid={sensor_humid}%")

import time
t0 = time.time()
result = pipeline.run(image, sensor_elevation_m=elev, sensor_temperature_c=sensor_temp, sensor_humidity_pct=sensor_humid)
dt = time.time() - t0

print(f"\n{'='*60}")
print(f"MACRO OUTPUT ({dt:.0f}s):")
print(result.stage_outputs.get("macro", ""))
