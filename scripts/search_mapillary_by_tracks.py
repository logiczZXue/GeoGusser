"""Query Mapillary for images along 2bulu hiking tracks.

For each track: compute bounding box → query Mapillary → download images with GPS.

Usage:
    python scripts/search_mapillary_by_tracks.py \
        --tracks data/2bulu_tracks.json \
        --token "MLY|xxx" \
        --out data/mapillary_trails/
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path


def get_track_bboxes(tracks_json: str, padding: float = 0.01) -> list:
    """Load tracks and compute bounding boxes. Returns [(name, west, south, east, north), ...]."""
    with open(tracks_json, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    # De-duplicate by removing (N) suffix
    unique = {}
    for k, v in tracks.items():
        base = re.sub(r'\s*\(\d+\)$', '', k)
        if base not in unique or v['num_points'] > unique[base]['num_points']:
            unique[base] = v

    bboxes = []
    for name, info in unique.items():
        lat_range = info['lat_range']
        lng_range = info['lng_range']
        west = max(-180, lng_range[0] - padding)
        south = max(-90, lat_range[0] - padding)
        east = min(180, lng_range[1] + padding)
        north = min(90, lat_range[1] + padding)
        bboxes.append((name, west, south, east, north))

    print(f"Loaded {len(bboxes)} unique tracks")
    return bboxes


def tile_large_bbox(west, south, east, north, tile_size=0.06):
    """Split bbox into tiles <= 0.0036 sq deg for Mapillary API."""
    tiles = []
    lat = south
    while lat < north:
        lng = west
        while lng < east:
            tiles.append((lng, lat, min(lng + tile_size, east), min(lat + tile_size, north)))
            lng += tile_size
        lat += tile_size
    return tiles


def fetch_images_in_bbox(token, west, south, east, north, limit=50):
    """Query Mapillary for images in a bounding box."""
    url = (
        f"https://graph.mapillary.com/images"
        f"?access_token={token}"
        f"&fields=id,computed_geometry,thumb_2048_url"
        f"&bbox={west},{south},{east},{north}"
        f"&limit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception:
        return [], None

    results = []
    for img in data.get("data", []):
        coords = img.get("computed_geometry", {}).get("coordinates", [0, 0])
        thumb = img.get("thumb_2048_url")
        if not thumb:
            continue
        results.append({
            "id": img["id"],
            "lat": coords[1],
            "lng": coords[0],
            "url": thumb,
        })
    return results, None


def download_image(img_info, output_dir, timeout=30):
    """Download a single image. Returns local path or None."""
    path = os.path.join(output_dir, f"{img_info['id']}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    try:
        req = urllib.request.Request(img_info["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 1000:
                return None
            with open(path, "wb") as f:
                f.write(data)
            return path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Search Mapillary along 2bulu tracks")
    parser.add_argument("--tracks", type=str, required=True, help="2bulu tracks JSON")
    parser.add_argument("--token", type=str, required=True, help="Mapillary API token")
    parser.add_argument("--out", type=str, default="data/mapillary_trails", help="Output directory")
    parser.add_argument("--padding", type=float, default=0.01, help="Padding around track bbox")
    parser.add_argument("--tile-size", type=float, default=0.06, help="Tile size in degrees")
    parser.add_argument("--dry-run", action="store_true", help="Only discover, don't download")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tracks = get_track_bboxes(args.tracks, padding=args.padding)

    # Phase 1: Discover images
    print(f"\n=== Phase 1: Discovering images along tracks ===")
    all_images = {}
    total_tiles = 0
    track_results = {}

    for name, west, south, east, north in tracks:
        tiles = tile_large_bbox(west, south, east, north, args.tile_size)
        track_imgs = 0
        for bbox in tiles:
            total_tiles += 1
            imgs, err = fetch_images_in_bbox(args.token, *bbox)
            if err:
                continue
            for img in (imgs or []):
                if img["id"] not in all_images:
                    all_images[img["id"]] = img
                    track_imgs += 1
            time.sleep(0.1)

        track_results[name] = track_imgs
        print(f"  {name[:40]}: {track_imgs} images (total unique: {len(all_images)})")

    print(f"\nTotal unique images: {len(all_images)}")
    print(f"Tiles queried: {total_tiles}")

    # Per-track summary
    for name, count in sorted(track_results.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count}")

    if args.dry_run:
        print("\n[Dry run - no downloads]")
        return

    # Phase 2: Download
    print(f"\n=== Phase 2: Downloading {len(all_images)} images ===")
    geotagged = {}
    downloaded = 0
    failed = 0
    t0 = time.time()

    imgs_list = list(all_images.values())
    for i, img in enumerate(imgs_list):
        path = download_image(img, args.out)
        if path:
            geotagged[path] = [img["lat"], img["lng"]]
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(imgs_list) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(imgs_list)}: {downloaded} ok, {failed} fail ({rate:.0f}/s, ETA {eta:.0f}s)")

        time.sleep(0.05)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s: {downloaded} downloaded, {failed} failed")

    # Save geotagged.json
    output_json = os.path.join(args.out, "geotagged.json")
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(geotagged)
        geotagged = existing

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(geotagged, f, indent=2, ensure_ascii=False)

    lats = [v[0] for v in geotagged.values()]
    lngs = [v[1] for v in geotagged.values()]
    if lats:
        print(f"\nTraining pairs saved: {output_json}")
        print(f"Total: {len(geotagged)}, Lat: [{min(lats):.4f},{max(lats):.4f}], Lng: [{min(lngs):.4f},{max(lngs):.4f}]")


if __name__ == "__main__":
    main()
