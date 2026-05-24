"""Generate prioritized Baidu Street View sampling points.

Prioritizes points near GeoKB compound scenes and landmarks for maximum
geographic feature diversity. Outputs a JSON file consumable by
download_baidu_panorama.py or manual review.

Usage:
    python scripts/generate_baidu_sample_points.py --dry-run    # preview plan
    python scripts/generate_baidu_sample_points.py --output data/baidu_sampling/priority_points.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.geo_kb import COMPOUND_SCENES


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── GeoKB landmark extraction ───────────────────────────────────────────────

def extract_geokb_landmarks() -> list[dict]:
    """Extract landmark anchor points from GeoKB compound scenes.

    Uses the first bbox of each compound scene as the anchor region.
    Returns list of {name, lat, lng, scene_type, category} dicts.
    """
    landmarks = []
    for name, config in COMPOUND_SCENES.items():
        bboxes = config.get("bboxes", [])
        if not bboxes:
            continue
        # Use the center of the first (primary) bbox
        bbox = bboxes[0]
        lat = (bbox.lat_min + bbox.lat_max) / 2
        lng = (bbox.lng_min + bbox.lng_max) / 2
        conditions = config.get("conditions", {})
        landmarks.append({
            "name": name,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "scene_type": conditions.get("scene_type", name),
            "category": conditions.get("terrain_type", "unknown"),
            "area_km2": bbox.area_km2,
        })
    return landmarks


# ── Sampling point generation ───────────────────────────────────────────────

def generate_landmark_proximal_points(
    landmarks: list[dict],
    radius_km: float = 2.0,
    points_per_landmark: int = 8,
) -> list[dict]:
    """Generate sampling points in a ring around each GeoKB landmark.

    Creates points at ~radius_km distance in 8 compass directions,
    plus the anchor point itself.
    """
    points = []
    headings = list(range(0, 360, 45))  # 8 headings

    for lm in landmarks:
        # The anchor point itself
        points.append({
            "lat": lm["lat"],
            "lng": lm["lng"],
            "priority": 0,
            "label": f"landmark_center:{lm['name']}",
            "scene_type": lm["scene_type"],
        })

        # Ring points at ~radius_km
        for heading in headings[:points_per_landmark]:
            lat, lng = _offset_point(lm["lat"], lm["lng"], radius_km, heading)
            if 17.0 <= lat <= 55.0 and 72.0 <= lng <= 136.0:
                points.append({
                    "lat": round(lat, 6),
                    "lng": round(lng, 6),
                    "priority": 0,
                    "label": f"landmark_ring:{lm['name']}:{heading}",
                    "scene_type": lm["scene_type"],
                })

    # Also add dense sampling within landmark bboxes (2x density of trails)
    for lm in landmarks:
        bbox_radius = math.sqrt(lm["area_km2"]) / 2 if lm["area_km2"] else 5.0
        # Sample every ~1km within the bbox centroid area
        spacing = 0.01  # ~1km
        for dlat in [-spacing, 0, spacing]:
            for dlng in [-spacing, 0, spacing]:
                if dlat == 0 and dlng == 0:
                    continue  # center already added
                lat = lm["lat"] + dlat
                lng = lm["lng"] + dlng
                if 17.0 <= lat <= 55.0 and 72.0 <= lng <= 136.0:
                    points.append({
                        "lat": round(lat, 6),
                        "lng": round(lng, 6),
                        "priority": 0,
                        "label": f"landmark_grid:{lm['name']}",
                        "scene_type": lm["scene_type"],
                    })

    return points


def _offset_point(lat: float, lng: float, dist_km: float, heading_deg: float):
    """Compute a point at given distance and heading from origin."""
    heading_rad = math.radians(heading_deg)
    dlat = dist_km * math.cos(heading_rad) / 111.0
    dlng = dist_km * math.sin(heading_rad) / (111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng


def generate_trail_points(
    trails_data: dict,
    spacing_km: float = 0.5,
    priority: int = 2,
) -> list[dict]:
    """Generate sampling points along hiking trails at ~spacing_km intervals."""
    points = []
    for trail_name, trail_info in trails_data.items():
        if trail_name.startswith("_"):
            continue
        raw = trail_info.get("trail_points", [])
        for i in range(len(raw) - 1):
            lat1, lng1 = raw[i]
            lat2, lng2 = raw[i + 1]
            dist = haversine_km(lat1, lng1, lat2, lng2)
            n_steps = max(1, int(dist / spacing_km))
            for step in range(n_steps):
                t = step / n_steps
                points.append({
                    "lat": round(lat1 + t * (lat2 - lat1), 6),
                    "lng": round(lng1 + t * (lng2 - lng1), 6),
                    "priority": priority,
                    "label": f"trail:{trail_name}",
                    "scene_type": "trail",
                })
        # Endpoint
        if raw:
            points.append({
                "lat": round(raw[-1][0], 6),
                "lng": round(raw[-1][1], 6),
                "priority": priority,
                "label": f"trail:{trail_name}",
                "scene_type": "trail",
            })
        # Nearby towns
        for tc in trail_info.get("town_coords", []):
            points.append({
                "lat": round(tc[0], 6),
                "lng": round(tc[1], 6),
                "priority": priority,
                "label": f"trail_town:{trail_name}",
                "scene_type": "town",
            })
    return points


def generate_city_grid_points(
    cities_data: dict,
    spacing_km: float = 1.5,
    priority: int = 3,
) -> list[dict]:
    """Generate grid sampling points within city bounding boxes.

    Uses ~spacing_km grid. For cities with distinctive architecture
    (Beijing, Shanghai, Lhasa, etc.), uses finer spacing.
    """
    HIGH_PRIORITY_CITIES = {
        "北京", "上海", "拉萨", "丽江", "大理", "成都", "西安",
        "哈尔滨", "乌鲁木齐", "桂林", "昆明", "苏州", "杭州",
        "广州", "厦门", "青岛",
    }

    points = []
    for city_name, city_info in cities_data.items():
        if city_name.startswith("_"):
            continue

        lat_min, lat_max = city_info["lat_range"]
        lng_min, lng_max = city_info["lng_range"]

        # Finer spacing for high-priority cities
        spacing = spacing_km * 0.7 if city_name in HIGH_PRIORITY_CITIES else spacing_km
        spacing_deg = spacing / 111.0  # approximate

        city_priority = 3 if city_name in HIGH_PRIORITY_CITIES else priority  # P3 vs P4

        lat = lat_min
        while lat <= lat_max:
            lng = lng_min
            while lng <= lng_max:
                points.append({
                    "lat": round(lat, 6),
                    "lng": round(lng, 6),
                    "priority": city_priority,
                    "label": f"city:{city_name}",
                    "scene_type": "city",
                })
                lng += spacing_deg
            lat += spacing_deg

        # Key areas (downtown landmarks)
        for area in city_info.get("key_areas", []):
            points.append({
                "lat": round(area[0], 6),
                "lng": round(area[1], 6),
                "priority": 1,
                "label": f"city_key:{city_name}",
                "scene_type": "city_key",
            })

    return points


# ── Deduplication ───────────────────────────────────────────────────────────

def deduplicate_points(points: list[dict], tolerance_km: float = 0.1) -> list[dict]:
    """Remove duplicate points within tolerance_km, keeping highest priority."""
    if not points:
        return points

    # Sort by priority (lower = more important)
    points.sort(key=lambda p: (p["priority"], p["label"]))

    kept = []
    for p in points:
        dup = False
        for k in kept:
            if haversine_km(p["lat"], p["lng"], k["lat"], k["lng"]) < tolerance_km:
                dup = True
                break
        if not dup:
            kept.append(p)

    return kept


def trim_proportional(points: list[dict], max_total: int) -> list[dict]:
    """Trim points to max_total while preserving proportional distribution.

    Allocation: 20% landmarks (P0), 20% trail (P1), 30% high-priority city (P3), 30% regular city (P4)
    """
    by_priority = defaultdict(list)
    for p in points:
        by_priority[p["priority"]].append(p)

    # Target allocations
    allocations = {
        0: 0.20,   # landmarks
        1: 0.05,   # city_key areas
        2: 0.20,   # trails
        3: 0.30,   # high-priority cities
        4: 0.25,   # regular cities
    }

    trimmed = []
    for pri, fraction in allocations.items():
        target_n = int(max_total * fraction)
        candidates = by_priority.get(pri, [])
        if len(candidates) > target_n:
            import random
            random.seed(42)
            candidates = random.sample(candidates, target_n)
        trimmed.extend(candidates)

    # Fill remaining quota from lowest priority
    remaining = max_total - len(trimmed)
    if remaining > 0:
        leftover = []
        for pri in sorted(by_priority.keys(), reverse=True):
            used = sum(1 for t in trimmed if t["priority"] == pri)
            available = by_priority[pri]
            if len(available) > used:
                import random
                random.seed(42)
                extra = random.sample(available, min(remaining, len(available) - used))
                leftover.extend(extra)
                remaining -= len(extra)
                if remaining <= 0:
                    break
        trimmed.extend(leftover)

    # Re-sort by priority
    trimmed.sort(key=lambda p: (p["priority"], p["label"]))
    return trimmed[:max_total]


# ── Report ──────────────────────────────────────────────────────────────────

def print_summary(points: list[dict]):
    """Print sampling plan summary."""
    by_priority = defaultdict(int)
    by_type = defaultdict(int)
    scene_types = defaultdict(int)

    for p in points:
        by_priority[p["priority"]] += 1
        label_type = p["label"].split(":")[0]
        by_type[label_type] += 1
        scene_types[p.get("scene_type", "unknown")] += 1

    print(f"\n{'='*60}")
    print(f"Sampling Plan Summary: {len(points)} points")
    print(f"  Daily quota: 1000 calls, ~{math.ceil(len(points)/1000)} days needed")
    print(f"\n  By priority:")
    for pri, count in sorted(by_priority.items()):
        print(f"    P{pri}: {count}")
    print(f"\n  By source:")
    for src, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    print(f"\n  By scene type:")
    for st, count in sorted(scene_types.items(), key=lambda x: -x[1]):
        print(f"    {st}: {count}")
    print(f"{'='*60}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate prioritized Baidu Street View sampling points")
    parser.add_argument("--output", type=str,
                        default="data/baidu_sampling/priority_points.json")
    parser.add_argument("--trails", type=str,
                        default="data/baidu_sampling/trails_coords.json")
    parser.add_argument("--cities", type=str,
                        default="data/baidu_sampling/cities_coords.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-points", type=int, default=7000,
                        help="Max total points (target: 7000 for 7-day quota)")
    args = parser.parse_args()

    # Load coordinate databases
    trails_data = {}
    if os.path.exists(args.trails):
        with open(args.trails, "r", encoding="utf-8") as f:
            trails_data = json.load(f)
        print(f"Trails: {sum(1 for k in trails_data if not k.startswith('_'))} routes")

    cities_data = {}
    if os.path.exists(args.cities):
        with open(args.cities, "r", encoding="utf-8") as f:
            cities_data = json.load(f)
        print(f"Cities: {sum(1 for k in cities_data if not k.startswith('_'))} regions")

    # ── Priority 1: GeoKB landmark-proximal points ─────────────────────
    landmarks = extract_geokb_landmarks()
    print(f"GeoKB landmarks: {len(landmarks)}")
    landmark_points = generate_landmark_proximal_points(
        landmarks, radius_km=2.0, points_per_landmark=8)
    print(f"  Landmark-proximal points: {len(landmark_points)}")

    # ── Priority 2: Trail points ───────────────────────────────────────
    trail_points = generate_trail_points(trails_data, spacing_km=0.5, priority=2)
    print(f"Trail points: {len(trail_points)}")

    # ── Priority 3: City grid points ───────────────────────────────────
    city_points = generate_city_grid_points(cities_data, spacing_km=1.5, priority=3)
    print(f"City grid points: {len(city_points)}")

    # ── Combine, deduplicate, trim ─────────────────────────────────────
    all_points = landmark_points + trail_points + city_points
    print(f"\nTotal raw points: {len(all_points)}")
    all_points = deduplicate_points(all_points, tolerance_km=0.05)
    print(f"After dedup: {len(all_points)}")

    # Trim with proportional allocation
    if args.max_points and len(all_points) > args.max_points:
        all_points = trim_proportional(all_points, args.max_points)
        print(f"Trimmed to {len(all_points)} (proportional allocation)")

    print_summary(all_points)

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_points, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")

    return all_points


if __name__ == "__main__":
    main()
