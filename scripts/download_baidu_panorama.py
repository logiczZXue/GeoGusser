"""Download Baidu Panorama static images with precise GPS for training.

Queries Baidu Panorama Static API at sampled GPS points along hiking trails
and city grids. Records exact GPS per image for regression head training.

API endpoints:
  - mapsv0.bdimg.com (no key needed, may have lower quality/availability)
  - api.map.baidu.com (requires AK, official API)

Usage:
    python scripts/download_baidu_panorama.py --mode trail  # trails only
    python scripts/download_baidu_panorama.py --mode city   # cities only
    python scripts/download_baidu_panorama.py --mode all    # both (default)
    python scripts/download_baidu_panorama.py --ak YOUR_AK  # with API key
    python scripts/download_baidu_panorama.py --dry-run     # print plan, no download
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "data/baidu_sampling/images"
MANIFEST_FILE = "data/baidu_sampling/baidu_geotagged.json"
TRAILS_FILE = "data/baidu_sampling/trails_coords.json"
CITIES_FILE = "data/baidu_sampling/cities_coords.json"

# Daily quota
DAILY_LIMIT = 1000

# Image params
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512
FOV = 90
HEADINGS = [0, 90, 180, 270]  # 4 directions per point (or 1 to save quota)

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests
RETRY_DELAY = 10     # seconds after rate limit

# Mapsv0 endpoint (no API key)
MAPSV0_URL = (
    "https://mapsv0.bdimg.com/?qt=pr3d"
    "&width={width}&height={height}"
    "&location={lng},{lat}"
    "&heading={heading}&pitch=0&fovy={fov}"
)


def sample_trail_points(trail_data: dict, spacing_km: float = 0.5) -> list:
    """Generate sampling points along a trail polyline.

    Interpolates between trail_points nodes to create dense sampling at
    ~spacing_km intervals. Also adds nearby town points.
    """
    points = []

    raw_points = trail_data.get("trail_points", [])
    # Interpolate between nodes
    for i in range(len(raw_points) - 1):
        lat1, lng1 = raw_points[i]
        lat2, lng2 = raw_points[i + 1]
        dist_km = haversine_km(lat1, lng1, lat2, lng2)
        num_steps = max(1, int(dist_km / spacing_km))
        for step in range(num_steps):
            t = step / num_steps
            lat = lat1 + t * (lat2 - lat1)
            lng = lng1 + t * (lng2 - lng1)
            points.append((lat, lng))

    # Add endpoint
    if raw_points:
        points.append(tuple(raw_points[-1]))

    # Add nearby towns
    for tc in trail_data.get("town_coords", []):
        points.append(tuple(tc))

    return points


def sample_city_grid(city_data: dict) -> list:
    """Generate grid sampling points within a city bounding box."""
    lat_min, lat_max = city_data["lat_range"]
    lng_min, lng_max = city_data["lng_range"]
    spacing = city_data.get("grid_spacing", 0.02)

    points = []
    lat = lat_min
    while lat <= lat_max:
        lng = lng_min
        while lng <= lng_max:
            points.append((round(lat, 6), round(lng, 6)))
            lng += spacing
        lat += spacing

    # Add key_areas if specified
    for area in city_data.get("key_areas", []):
        points.append(tuple(area))

    return points


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two GPS points in km."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def generate_url(lat: float, lng: float, heading: int,
                 use_mapsv0: bool = True) -> str:
    """Generate Baidu panorama static image URL."""
    if use_mapsv0:
        return MAPSV0_URL.format(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            lng=lng, lat=lat, heading=heading, fov=FOV,
        )
    else:
        # Placeholder for official API (requires AK)
        return ""


def download_image(url: str, dest: str, max_retries: int = 3) -> bool:
    """Download image with retry for rate limiting."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://map.baidu.com/",
    })
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                # Check for empty/error response (Baidu returns small error images)
                if len(data) < 1000:
                    return False
                with open(dest, "wb") as f:
                    f.write(data)
                return True
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Rate limited (429), retry in {wait}s...")
                time.sleep(wait)
                continue
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return False
    return False


def load_existing_manifest(manifest_path: str) -> dict:
    """Load existing download manifest for resumption."""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def point_already_downloaded(lat: float, lng: float, manifest: dict,
                              tolerance: float = 0.0005) -> bool:
    """Check if a nearby point was already downloaded (~50m tolerance)."""
    for key in manifest:
        parts = key.split(",")
        if len(parts) >= 2:
            try:
                mlat = float(parts[0])
                mlng = float(parts[1])
                if abs(mlat - lat) < tolerance and abs(mlng - lng) < tolerance:
                    return True
            except ValueError:
                continue
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Baidu Panorama static images for geolocation training"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["trail", "city", "all"])
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--trails", type=str, default=TRAILS_FILE)
    parser.add_argument("--cities", type=str, default=CITIES_FILE)
    parser.add_argument("--ak", type=str, default=None,
                        help="Baidu API key (for official API)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to download (0 = no limit)")
    parser.add_argument("--headings", type=int, default=4,
                        help="Headings per point: 1 (single) or 4 (all directions)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without downloading")
    parser.add_argument("--use-mapsv0", action="store_true", default=True,
                        help="Use mapsv0 endpoint (no API key needed)")
    args = parser.parse_args()

    use_mapsv0 = args.use_mapsv0 and not args.ak  # fallback if no AK

    # ── Load coordinate databases ─────────────────────────────────────
    trails_data = {}
    cities_data = {}

    if args.mode in ("trail", "all"):
        if os.path.exists(args.trails):
            with open(args.trails, "r", encoding="utf-8") as f:
                trails_data = json.load(f)
            # Remove metadata keys
            trails_data = {k: v for k, v in trails_data.items()
                           if not k.startswith("_")}
            print(f"Trails loaded: {len(trails_data)} routes")

    if args.mode in ("city", "all"):
        if os.path.exists(args.cities):
            with open(args.cities, "r", encoding="utf-8") as f:
                cities_data = json.load(f)
            cities_data = {k: v for k, v in cities_data.items()
                           if not k.startswith("_")}
            print(f"Cities loaded: {len(cities_data)} regions")

    # ── Generate sampling plan ─────────────────────────────────────────
    plan = {}  # {unique_key: (lat, lng, heading, label)}

    # Trail sampling
    for trail_name, trail_info in trails_data.items():
        trail_points = sample_trail_points(trail_info)
        for lat, lng in trail_points:
            headings = HEADINGS[:args.headings]
            for heading in headings:
                key = f"{lat:.6f},{lng:.6f},{heading}"
                plan[key] = (lat, lng, heading, f"trail:{trail_name}")

    # City grid sampling
    for city_name, city_info in cities_data.items():
        city_points = sample_city_grid(city_info)
        for lat, lng in city_points:
            headings = HEADINGS[:args.headings]
            for heading in headings:
                key = f"{lat:.6f},{lng:.6f},{heading}"
                plan[key] = (lat, lng, heading, f"city:{city_name}")

    print(f"\nSampling plan: {len(plan)} points ({len(trails_data)} trails, "
          f"{len(cities_data)} cities)")

    # Estimate quota usage
    est_days = math.ceil(len(plan) / DAILY_LIMIT)
    print(f"Estimated quota: {len(plan)} requests (~{est_days} days at {DAILY_LIMIT}/day)")

    if args.limit > 0:
        # Take first N from plan
        plan_keys = list(plan.keys())[:args.limit]
        plan = {k: plan[k] for k in plan_keys}
        print(f"Limited to {args.limit} requests")

    if args.dry_run:
        # Show breakdown
        trail_count = sum(1 for v in plan.values() if v[3].startswith("trail:"))
        city_count = sum(1 for v in plan.values() if v[3].startswith("city:"))
        print(f"\n[Dry-run] Would download: {trail_count} trail + {city_count} city images")
        print(f"[Dry-run] Output dir: {args.output_dir}")
        print(f"[Dry-run] Manifest: {MANIFEST_FILE}")
        return

    # ── Setup output ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_existing_manifest(MANIFEST_FILE)

    # ── Download loop ──────────────────────────────────────────────────
    downloaded = 0
    skipped = 0
    failed = 0
    daily_count = 0

    total = len(plan)
    print(f"\nDownloading {total} images to {args.output_dir}...")
    print(f"Existing manifest: {len(manifest)} entries")
    print(f"API endpoint: {'mapsv0 (no key)' if use_mapsv0 else 'official (AK)'}")

    for i, (key, (lat, lng, heading, label)) in enumerate(plan.items()):
        # Check daily quota
        if daily_count >= DAILY_LIMIT:
            print(f"\n[QUOTA] Daily limit reached ({DAILY_LIMIT}). "
                  f"Resume tomorrow from index {i}.")
            break

        # Skip existing
        if key in manifest:
            skipped += 1
            continue

        # Build URL
        url = generate_url(lat, lng, heading, use_mapsv0=use_mapsv0)
        if not url and not use_mapsv0:
            print(f"\n[ERROR] Official API not yet implemented. Use --use-mapsv0 or provide --ak")
            break

        # Download
        filename = f"bd_{lat:.6f}_{lng:.6f}_{heading}.jpg"
        dest = os.path.join(args.output_dir, filename)

        success = download_image(url, dest, max_retries=3)
        daily_count += 1

        if success:
            downloaded += 1
            manifest[key] = [lat, lng, heading, label]
        else:
            failed += 1
            # Remove empty file
            if os.path.exists(dest) and os.path.getsize(dest) < 1000:
                os.remove(dest)

        # Progress
        pct = (i + 1) / total * 100
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {pct:.1f}% | "
                  f"OK={downloaded} skip={skipped} fail={failed} | "
                  f"daily={daily_count}/{DAILY_LIMIT}")

        # Rate limit
        time.sleep(REQUEST_DELAY)

        # Save manifest periodically
        if (i + 1) % 100 == 0:
            with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"    [Saved manifest: {len(manifest)} entries]")

    # ── Final save ─────────────────────────────────────────────────────
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    print(f"  Manifest:   {MANIFEST_FILE} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
