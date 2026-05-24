"""
Filter fg_geo YFCC geotagged subset for China and download images.

Source: https://github.com/visipedia/fg_geo
File: yfcc100m_geolocated_inat2017species.csv.zip (2.7 MB, 36,144 records)
Fields: yfcc100m ID, line_number, label, latitude, longitude, Flickr URL, mmcommons URL
"""

import csv
import io
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

CHINA_BBOX = {"lon_min": 73.66, "lon_max": 135.04, "lat_min": 3.86, "lat_max": 53.55}
DATA_URL = "https://github.com/visipedia/fg_geo/raw/master/data/yfcc100m_geolocated_inat2017species.csv.zip"
OUT_DIR = Path("D:/Geocomp/data/yfcc_china")


def in_china(lat, lon):
    return (CHINA_BBOX["lon_min"] <= lon <= CHINA_BBOX["lon_max"] and
            CHINA_BBOX["lat_min"] <= lat <= CHINA_BBOX["lat_max"])


def download_image(url, save_path, max_retries=2):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "YFCC-DL/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) < 1000:
                    continue
                save_path.write_bytes(data)
                return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and parse CSV
    print("Downloading fg_geo CSV (2.7 MB)...")
    resp = urllib.request.urlopen(DATA_URL)
    raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv_data = z.read("yfcc100m_geolocated_inat2017species.csv")

    # Handle encoding
    reader = csv.DictReader(io.StringIO(csv_data.decode("latin-1")))
    all_rows = list(reader)
    print(f"Total records: {len(all_rows):,}")

    # 2. Filter for China
    china_rows = []
    for row in all_rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (ValueError, KeyError):
            continue
        if in_china(lat, lon):
            china_rows.append(row)

    print(f"In China bbox: {len(china_rows):,}")

    # Save metadata
    meta_file = OUT_DIR / "yfcc_china_metadata.csv"
    with open(meta_file, "w", encoding="utf-8", newline="") as f:
        if china_rows:
            writer = csv.DictWriter(f, fieldnames=china_rows[0].keys())
            writer.writeheader()
            writer.writerows(china_rows)
    print(f"Metadata saved: {meta_file}")

    # 3. Download images
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    to_download = china_rows[:5000]  # cap

    downloaded = 0
    failed = 0
    t0 = time.time()

    for i, row in enumerate(to_download):
        flickr_url = row.get("Flickr URL", "").strip()
        s3_url = row.get("mmcommons URL", "").strip()
        yfcc_id = row.get("yfcc100m ID", f"img_{i}")

        fname = f"{yfcc_id}.jpg"
        fpath = img_dir / fname
        if fpath.exists():
            downloaded += 1
            continue

        saved = False
        for url in [s3_url, flickr_url]:
            if url and download_image(url, fpath):
                saved = True
                break

        if saved:
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(to_download)} ({rate:.1f}/s) done:{downloaded} fail:{failed}")

    elapsed = time.time() - t0
    print(f"\nDone. Downloaded: {downloaded}, Failed: {failed}, Time: {elapsed:.0f}s")
    print(f"Images: {img_dir}")


if __name__ == "__main__":
    main()
