"""GeoDecoder: Data preparation tool for baseline experiments.

Downloads street view panorama images from the tuxun dataset and builds
a metadata index for geolocation experiments. As the original author noted,
"decoder的代码是复现baseline的数据集基础".

Usage:
    python -m src.baseline.Geodecoder --countries France,Japan --output-dir data/panoramas
    python -m src.baseline.Geodecoder --from-csv panoids.csv --output-dir data/panoramas
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, get_path


# Default target countries for the competition
DEFAULT_COUNTRIES = [
    "France", "Japan", "Thailand", "Brazil", "United States",
    "Germany", "India", "South Korea", "Australia", "Nigeria",
]


def extract_panoids_from_dataset(
    dataset_path: str,
    target_countries: list[str],
    max_per_country: int = 160,
) -> dict[str, list[str]]:
    """Extract panorama IDs from the tuxun dataset for target countries.

    Reuses the pattern from PanoIDCollector.py.
    """
    csv.field_size_limit(sys.maxsize)
    country_panoids = defaultdict(list)
    remaining = set(target_countries)

    with open(dataset_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
        f.seek(0)
        reader = csv.DictReader(f)
        from tqdm import tqdm

        for row in tqdm(reader, total=total - 1, desc="Scanning dataset"):
            if not remaining:
                break
            try:
                data = json.loads(row.get("data", "{}"))
                rounds_data = data.get("rounds", [])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            for rd in rounds_data:
                if not isinstance(rd, dict):
                    continue
                country = rd.get("nation", "")
                panoid = rd.get("panoId", "")
                if country in remaining and len(country_panoids[country]) < max_per_country:
                    country_panoids[country].append(panoid)
                    if len(country_panoids[country]) >= max_per_country:
                        remaining.discard(country)

    return dict(country_panoids)


def download_panorama_images(
    panoids: dict[str, list[str]],
    output_dir: str,
    api_key: str = "",
):
    """Download panorama images using Google Street View Static API.

    Args:
        panoids: Dict mapping country -> list of panorama IDs.
        output_dir: Directory to save images.
        api_key: Google Maps API key.
    """
    import requests

    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://maps.googleapis.com/maps/api/streetview"

    for country, pids in panoids.items():
        country_dir = os.path.join(output_dir, country)
        os.makedirs(country_dir, exist_ok=True)

        for panoid in pids:
            # Save 4 directions + panoramic
            for direction in ["north", "east", "south", "west", "panoramic"]:
                out_path = os.path.join(country_dir, f"{panoid}_{direction}.jpg")
                if os.path.exists(out_path):
                    continue

                heading = {"north": 0, "east": 90, "south": 180, "west": 270, "panoramic": 180}
                fov = 120 if direction == "panoramic" else 90

                params = {
                    "size": "640x640",
                    "pano": panoid,
                    "heading": heading[direction],
                    "fov": fov,
                    "pitch": 0,
                    "key": api_key,
                }

                try:
                    resp = requests.get(base_url, params=params, timeout=30)
                    if resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  Failed: {panoid}_{direction}: {e}")

        print(f"  {country}: {len(pids)} panoramas")


def build_metadata_index(panoids: dict[str, list[str]], output_dir: str):
    """Build a metadata index CSV: panoid, country, direction, filepath."""
    index_path = os.path.join(output_dir, "metadata_index.csv")
    with open(index_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panoid", "country", "direction", "filepath"])
        for country, pids in panoids.items():
            for pid in pids:
                for direction in ["north", "east", "south", "west", "panoramic"]:
                    filepath = os.path.join(country, f"{pid}_{direction}.jpg")
                    writer.writerow([pid, country, direction, filepath])
    print(f"Metadata index saved to {index_path}")


def main():
    parser = argparse.ArgumentParser(description="GeoDecoder: Download street view data for baseline experiments")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated country names")
    parser.add_argument("--max-per-country", type=int, default=160)
    parser.add_argument("--output-dir", type=str, default="data/panoramas")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to tuxun_combined.csv")
    parser.add_argument("--api-key", type=str, default=None, help="Google Maps API key")
    parser.add_argument("--dry-run", action="store_true", help="Only extract panoids, don't download")
    args = parser.parse_args()

    countries = args.countries.split(",") if args.countries else DEFAULT_COUNTRIES
    api_key = args.api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")

    # Find dataset
    dataset_path = args.dataset_path
    if not dataset_path:
        cfg = load_config()
        data_dir = get_path(cfg, "paths", "data_dir")
        dataset_path = os.path.join(data_dir, "tuxun_combined.csv")
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Please provide --dataset-path or place tuxun_combined.csv in data/")
            return

    print(f"Extracting panorama IDs for {len(countries)} countries...")
    panoids = extract_panoids_from_dataset(dataset_path, countries, args.max_per_country)

    total = sum(len(v) for v in panoids.values())
    for c, pids in panoids.items():
        print(f"  {c}: {len(pids)} panoramas")
    print(f"Total: {total} panoramas")

    build_metadata_index(panoids, args.output_dir)

    if not args.dry_run:
        if not api_key:
            print("Warning: No Google Maps API key. Set GOOGLE_MAPS_API_KEY or pass --api-key")
            print("Skipping image download. Use --dry-run to only extract panoids.")
            return
        download_panorama_images(panoids, args.output_dir, api_key)


if __name__ == "__main__":
    main()
