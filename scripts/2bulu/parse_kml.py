"""Parse downloaded KML/GPX files to extract GPS coordinate sequences.

Output: JSON mapping track_name → list of [lat, lng, elevation] points

Usage:
    python scripts/2bulu/parse_kml.py --kml-dir data/2bulu_kml/ --out data/2bulu_tracks.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_kml_file(filepath: str) -> list:
    """Extract coordinate points from a KML file. Returns [(lat, lng, ele), ...]."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return []

    # KML namespace
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    # Also try without namespace
    namespaces = [
        {"kml": "http://www.opengis.net/kml/2.2"},
        {"kml": "http://earth.google.com/kml/2.2"},
        {"kml": "http://earth.google.com/kml/2.1"},
        {"kml": "http://earth.google.com/kml/2.0"},
        {},
    ]

    points = []
    for ns_map in namespaces:
        if ns_map:
            coords_elems = root.findall(".//kml:coordinates", ns_map)
        else:
            coords_elems = root.findall(".//{http://www.opengis.net/kml/2.2}coordinates")
            if not coords_elems:
                coords_elems = root.findall(".//{http://earth.google.com/kml/2.2}coordinates")
            if not coords_elems:
                coords_elems = root.findall(".//coordinates")

        if coords_elems:
            break

    for elem in coords_elems:
        text = elem.text.strip() if elem.text else ""
        for line in text.split():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    lng, lat = float(parts[0]), float(parts[1])
                    ele = float(parts[2]) if len(parts) >= 3 else 0.0
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        points.append([lat, lng, ele])
                except ValueError:
                    continue

    return points


def parse_gpx_file(filepath: str) -> list:
    """Extract track points from a GPX file."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return []

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    points = []

    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.get("lat", 0))
        lng = float(trkpt.get("lon", 0))
        ele_elem = trkpt.find("gpx:ele", ns)
        ele = float(ele_elem.text) if ele_elem is not None and ele_elem.text else 0.0
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            points.append([lat, lng, ele])

    if not points:
        # Try without namespace
        for trkpt in root.findall(".//{http://www.topografix.com/GPX/1/1}trkpt"):
            lat = float(trkpt.get("lat", 0))
            lng = float(trkpt.get("lon", 0))
            points.append([lat, lng, 0.0])

    return points


def main():
    parser = argparse.ArgumentParser(description="Parse KML/GPX files to coordinates")
    parser.add_argument("--kml-dir", type=str, required=True, help="Directory with KML/GPX files")
    parser.add_argument("--out", type=str, default="data/2bulu_tracks.json", help="Output JSON file")
    args = parser.parse_args()

    if not os.path.isdir(args.kml_dir):
        print(f"Directory not found: {args.kml_dir}")
        sys.exit(1)

    all_tracks = {}
    total_points = 0
    files_processed = 0

    for fname in sorted(os.listdir(args.kml_dir)):
        fpath = os.path.join(args.kml_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".kml", ".gpx", ".xml"):
            continue

        if ext == ".gpx":
            points = parse_gpx_file(fpath)
        else:
            points = parse_kml_file(fpath)

        if points:
            track_name = os.path.splitext(fname)[0]
            all_tracks[track_name] = {
                "file": fname,
                "num_points": len(points),
                "points": points,
                "lat_range": [min(p[0] for p in points), max(p[0] for p in points)],
                "lng_range": [min(p[1] for p in points), max(p[1] for p in points)],
                "ele_range": [min(p[2] for p in points), max(p[2] for p in points)] if points[0][2] else None,
            }
            total_points += len(points)
            files_processed += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_tracks, f, indent=2, ensure_ascii=False)

    print(f"Parsed {files_processed} files, {total_points} total track points")
    print(f"Saved to {args.out}")

    # Print summary per track
    for name, info in all_tracks.items():
        print(f"  {name}: {info['num_points']} pts, "
              f"lat[{info['lat_range'][0]:.4f},{info['lat_range'][1]:.4f}], "
              f"lng[{info['lng_range'][0]:.4f},{info['lng_range'][1]:.4f}]")


if __name__ == "__main__":
    main()
