"""Download geotagged test images from Wikimedia Commons for worldwide coverage.

Usage:
    python scripts/download_test_images.py [--output-dir data/test_images/] [--count 20]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Worldwide locations for diverse coverage (lat, lng, radius_km, continent)
LOCATIONS = [
    (48.8566, 2.3522, 5000, "Europe", "Paris"),
    (51.5074, -0.1278, 5000, "Europe", "London"),
    (52.5200, 13.4050, 5000, "Europe", "Berlin"),
    (35.6762, 139.6503, 5000, "Asia", "Tokyo"),
    (31.2304, 121.4737, 5000, "Asia", "Shanghai"),
    (13.7563, 100.5018, 5000, "Asia", "Bangkok"),
    (28.6139, 77.2090, 5000, "Asia", "Delhi"),
    (40.7128, -74.0060, 5000, "North America", "NewYork"),
    (34.0522, -118.2437, 5000, "North America", "LosAngeles"),
    (19.4326, -99.1332, 5000, "North America", "MexicoCity"),
    (-23.5505, -46.6333, 5000, "South America", "SaoPaulo"),
    (-34.6037, -58.3816, 5000, "South America", "BuenosAires"),
    (-33.9249, 18.4241, 5000, "Africa", "CapeTown"),
    (-1.2921, 36.8219, 5000, "Africa", "Nairobi"),
    (30.0444, 31.2357, 5000, "Africa", "Cairo"),
    (-33.8688, 151.2093, 5000, "Oceania", "Sydney"),
    (-41.2865, 174.7762, 5000, "Oceania", "Wellington"),
    (55.7558, 37.6173, 5000, "Europe", "Moscow"),
    (37.7749, -122.4194, 5000, "North America", "SanFrancisco"),
    (25.2048, 55.2708, 5000, "Asia", "Dubai"),
    (59.9139, 10.7522, 5000, "Europe", "Oslo"),
    (14.5995, 120.9842, 5000, "Asia", "Manila"),
    (-22.9068, -43.1729, 5000, "South America", "Rio"),
    (6.5244, 3.3792, 5000, "Africa", "Lagos"),
]

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Geocomp/1.0 (research project; contact@example.com)"
DOWNLOAD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQUEST_DELAY = 3.0  # seconds between API calls
DOWNLOAD_DELAY = 5.0  # seconds between image downloads (avoid 429 rate limit)


def search_location(lat: float, lng: float, radius: int = 5000, limit: int = 5) -> list[dict]:
    """Search for geotagged images near a location."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lng}",
        "ggsradius": str(radius),
        "ggslimit": str(limit),
        "ggsnamespace": "6",  # File namespace only
        "prop": "imageinfo|coordinates",
        "iiprop": "url|extmetadata",
        "coprimary": "primary",
    }
    qs = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    url = f"{API_URL}?{qs}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}")
        return []

    results = []
    pages = (data.get("query") or {}).get("pages", {})
    for pid, page in pages.items():
        imageinfo = (page.get("imageinfo") or [{}])[0]
        image_url = imageinfo.get("url", "")
        desc_url = imageinfo.get("descriptionurl", "")

        # Filter out non-photo files (SVG, icons, diagrams)
        title = page.get("title", "")
        ext = os.path.splitext(title)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue

        # Get coordinates from the page
        coords = None
        coordinates_list = page.get("coordinates", [])
        if coordinates_list:
            coords = (coordinates_list[0].get("lat"), coordinates_list[0].get("lon"))

        if image_url and coords:
            attrs = imageinfo.get("extmetadata", {})
            results.append({
                "title": title,
                "image_url": image_url,
                "desc_url": desc_url,
                "lat": coords[0],
                "lng": coords[1],
                "artist": attrs.get("Artist", {}).get("value", "Unknown"),
                "license": attrs.get("LicenseShortName", {}).get("value", "Unknown"),
            })
    return results


def download_image(url: str, dest: str, max_retries: int = 3) -> bool:
    """Download an image from URL to local path, with retry for rate limits."""
    req = urllib.request.Request(url, headers={
        "User-Agent": DOWNLOAD_UA,
        "Referer": "https://commons.wikimedia.org/",
    })
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"    Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"    Download failed: {e}")
            return False
        except Exception as e:
            print(f"    Download failed: {e}")
            return False
    return False


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Download geotagged test images")
    parser.add_argument("--output-dir", type=str, default="data/test_images")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    downloaded = 0
    needed = args.count

    print(f"Searching {len(LOCATIONS)} locations worldwide for geotagged images...")
    print(f"Need {needed} images.\n")

    for lat, lng, radius, continent, city in LOCATIONS:
        if downloaded >= needed:
            break

        print(f"  {city} ({continent}): searching...")
        results = search_location(lat, lng, radius, limit=5)
        time.sleep(REQUEST_DELAY)

        if not results:
            print(f"    No geotagged photos found")
            continue

        # Take only the first usable result per location (avoid rate limits)
        for r in results:
            if downloaded >= needed:
                break

            safe_name = f"{downloaded:02d}_{continent}_{city}.jpg"
            dest = out_dir / safe_name

            print(f"    Downloading: {r['title'][:60]}...")
            print(f"      Coords: {r['lat']:.4f}, {r['lng']:.4f}")
            print(f"      URL: {r['image_url'][:80]}...")

            if download_image(r["image_url"], str(dest)):
                file_size = dest.stat().st_size
                print(f"      OK ({file_size:,} bytes)")

                manifest.append({
                    "filename": safe_name,
                    "continent": continent,
                    "query_city": city,
                    "lat": r["lat"],
                    "lng": r["lng"],
                    "source_title": r["title"],
                    "source_url": r["desc_url"],
                    "artist": r["artist"],
                    "license": r["license"],
                })
                downloaded += 1
                time.sleep(DOWNLOAD_DELAY)
                break  # Only 1 image per location
            else:
                print(f"      SKIPPED (download failed)")
                time.sleep(DOWNLOAD_DELAY)

    # Save manifest
    manifest_path = out_dir / "ground_truth.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {downloaded}/{needed} images downloaded")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Images saved to: {out_dir}/")


if __name__ == "__main__":
    main()
