"""Download Mapillary images with GPS for training coordinate regression head.

Targets both urban streets AND wilderness hiking trails — matching the
"徒步眼镜" (hiking glasses) product requirement.

Usage:
    python scripts/download_mapillary.py --token "MLY|xxx" --regions china_hiking
    python scripts/download_mapillary.py --token "MLY|xxx" --regions all --limit 10000
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Target regions — mix of urban streets AND wilderness hiking areas
# ---------------------------------------------------------------------------

# Format: (name, west, south, east, north, priority)
# Priority 1 = hiking/outdoor, 2 = urban, 3 = mixed

_REGIONS = {
    # === Hiking / Wilderness (Priority 1) ===
    "huangshan":       ("黄山",        118.10, 30.05, 118.25, 30.20, 1),
    "zhangjiajie":     ("张家界",      110.35, 29.25, 110.50, 29.40, 1),
    "yulong":          ("玉龙雪山",    100.15, 27.05, 100.30, 27.20, 1),
    "wuyishan":        ("武夷山",      117.60, 27.55, 117.75, 27.70, 1),
    "taishan":         ("泰山",        117.00, 36.20, 117.15, 36.35, 1),
    "lushan":          ("庐山",        115.90, 29.45, 116.05, 29.60, 1),
    "emeishan":        ("峨眉山",      103.30, 29.50, 103.45, 29.65, 1),
    "huashan":         ("华山",        110.00, 34.40, 110.15, 34.55, 1),
    "guilin_yangshuo": ("桂林阳朔",    110.30, 24.95, 110.50, 25.10, 1),
    "taihang":         ("太行山大峡谷", 113.70, 35.80, 113.90, 35.95, 1),
    "siguniang":       ("四姑娘山",    102.80, 31.00, 102.95, 31.15, 1),
    "daocheng":        ("稻城亚丁",    100.30, 28.35, 100.45, 28.50, 1),
    "xishuangbanna":   ("西双版纳",    100.70, 21.90, 100.85, 22.05, 1),
    "tianshan":        ("天山天池",    88.00, 43.80, 88.15, 43.95, 1),
    "kanasi":          ("喀纳斯",      87.00, 48.60, 87.15, 48.75, 1),
    "changbaishan":    ("长白山",      128.00, 42.00, 128.15, 42.15, 1),

    # === Urban (Priority 2) ===
    "beijing":         ("北京城区",    116.20, 39.80, 116.60, 40.10, 2),
    "shanghai":        ("上海城区",    121.30, 31.10, 121.70, 31.40, 2),
    "guangzhou":       ("广州城区",    113.15, 23.00, 113.55, 23.25, 2),
    "shenzhen":        ("深圳城区",    113.85, 22.45, 114.25, 22.70, 2),
    "chengdu":         ("成都城区",    103.95, 30.55, 104.25, 30.80, 2),
    "chongqing":       ("重庆城区",    106.40, 29.45, 106.70, 29.70, 2),
    "kunming":         ("昆明城区",    102.60, 24.95, 102.90, 25.20, 2),
    "xian":            ("西安城区",    108.85, 34.15, 109.15, 34.40, 2),
    "hangzhou":        ("杭州城区",    120.00, 30.15, 120.30, 30.40, 2),
    "nanjing":         ("南京城区",    118.70, 31.95, 119.00, 32.20, 2),

    # === Scenic mountain roads (Priority 3) ===
    "western_sichuan": ("川西山区公路", 102.00, 30.50, 102.50, 31.00, 3),
    "yunnan_highway":  ("云南山区公路", 100.00, 26.00, 100.50, 26.50, 3),
    "guizhou_mountain":("贵州山区公路", 106.50, 26.50, 107.00, 27.00, 3),
}

# Tile size — must keep bbox area <= 0.01 sq degrees
TILE_SIZE = 0.06  # degrees, area = 0.0036 sq degrees (well under limit)
MAX_PER_TILE = 50  # Mapillary API limit per query


def tile_bbox(west, south, east, north, tile_size=TILE_SIZE):
    """Split a bounding box into tiles small enough for Mapillary API."""
    tiles = []
    lat = south
    while lat < north:
        lng = west
        while lng < east:
            tiles.append((lng, lat, min(lng + tile_size, east), min(lat + tile_size, north)))
            lng += tile_size
        lat += tile_size
    return tiles


def fetch_tile_images(token, bbox, limit=MAX_PER_TILE):
    """Fetch image metadata for one tile. Returns list of dicts with id, lat, lng, url."""
    west, south, east, north = bbox
    url = (
        f"https://graph.mapillary.com/images"
        f"?access_token={token}"
        f"&fields=id,computed_geometry,thumb_2048_url,width,height"
        f"&bbox={west},{south},{east},{north}"
        f"&limit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        msg = body.get("error", {}).get("message", str(e))
        if "too large" in msg.lower():
            return None, msg
        print(f"  HTTP {e.code}: {msg[:120]}")
        return [], None
    except Exception as e:
        return [], str(e)

    results = []
    for img in data.get("data", []):
        coords = img.get("computed_geometry", {}).get("coordinates", [0, 0])
        thumb_url = img.get("thumb_2048_url") or img.get("thumb_1024_url")
        if not thumb_url:
            continue
        results.append({
            "id": img["id"],
            "lat": coords[1],
            "lng": coords[0],
            "url": thumb_url,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
        })
    return results, None


def download_image(img_info, output_dir, timeout=30):
    """Download a single image, save as {id}.jpg. Returns local path or None."""
    img_path = os.path.join(output_dir, f"{img_info['id']}.jpg")
    if os.path.exists(img_path):
        return img_path  # Already downloaded

    try:
        req = urllib.request.Request(img_info["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 1000:
                return None
            with open(img_path, "wb") as f:
                f.write(data)
            return img_path
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Download Mapillary images for training")
    parser.add_argument("--token", type=str, required=True, help="Mapillary API token (MLY|...)")
    parser.add_argument("--regions", type=str, default="china_hiking",
                        choices=["china_hiking", "urban", "all"],
                        help="Which region set to download")
    parser.add_argument("--output", type=str, default="data/mapillary",
                        help="Output directory for images")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max total images (0=unlimited)")
    parser.add_argument("--tile-size", type=float, default=TILE_SIZE,
                        help="Tile size in degrees (default 0.06)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count available images without downloading")
    args = parser.parse_args()

    # Select regions by priority
    priorities = {"china_hiking": {1}, "urban": {2}, "all": {1, 2, 3}}
    selected = {k: v for k, v in _REGIONS.items()
                if v[5] in priorities[args.regions]}

    print(f"Regions: {len(selected)}")
    for key, (name, w, s, e, n, pri) in selected.items():
        num_tiles = len(tile_bbox(w, s, e, n, args.tile_size))
        print(f"  {key}: {name} ({num_tiles} tiles)")

    os.makedirs(args.output, exist_ok=True)

    # Phase 1: Discover all images
    print(f"\n=== Phase 1: Discovering images ===")
    all_images = {}  # id -> info (deduplicate)
    total_tiles = 0
    for key, (name, west, south, east, north, pri) in selected.items():
        tiles = tile_bbox(west, south, east, north, args.tile_size)
        region_count = 0
        for bbox in tiles:
            total_tiles += 1
            imgs, err = fetch_tile_images(args.token, bbox, MAX_PER_TILE)
            if err:
                continue
            for img in (imgs or []):
                if img["id"] not in all_images:
                    all_images[img["id"]] = img
                    region_count += 1
            time.sleep(0.1)  # Rate limiting

        print(f"  {key} ({name}): found {region_count} unique images "
              f"(total unique: {len(all_images)})")

        if args.limit and len(all_images) >= args.limit:
            break

    print(f"\nTotal unique images discovered: {len(all_images)}")
    print(f"Tiles queried: {total_tiles}")

    if args.dry_run:
        return

    # Phase 2: Download images
    print(f"\n=== Phase 2: Downloading images ===")
    data_json = {}
    downloaded = 0
    skipped = 0
    failed = 0
    imgs_to_dl = list(all_images.values())
    if args.limit:
        imgs_to_dl = imgs_to_dl[:args.limit]

    t0 = time.time()
    for i, img in enumerate(imgs_to_dl):
        path = download_image(img, args.output)
        if path:
            data_json[path] = [img["lat"], img["lng"]]
            if path and os.path.getsize(path) > 1000:
                downloaded += 1
            else:
                failed += 1
                data_json.pop(path, None)
        else:
            # Already exists
            skipped += 1
            data_json[os.path.join(args.output, f"{img['id']}.jpg")] = [img["lat"], img["lng"]]

        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(imgs_to_dl) - i - 1) / rate
            print(f"  {i+1}/{len(imgs_to_dl)}: {downloaded} new, {skipped} cached, "
                  f"{failed} failed ({rate:.0f}/s, ETA {eta:.0f}s)")

        time.sleep(0.05)  # Polite rate limiting

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s: {downloaded} downloaded, {skipped} cached, {failed} failed")

    # Phase 3: Save training data JSON
    output_json = os.path.join(args.output, "geotagged.json")
    # Merge with existing if any
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(data_json)
        data_json = existing

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_json, f, indent=2, ensure_ascii=False)

    lats = [v[0] for v in data_json.values()]
    lngs = [v[1] for v in data_json.values()]
    print(f"\nTraining data saved: {output_json}")
    print(f"Total samples: {len(data_json)}")
    print(f"Lat range: [{min(lats):.4f}, {max(lats):.4f}]")
    print(f"Lng range: [{min(lngs):.4f}, {max(lngs):.4f}]")


if __name__ == "__main__":
    main()
