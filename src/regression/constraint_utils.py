"""Spatial constraint utilities: bbox operations, constraint power, explanations.

Shared geometry helpers for the sensor-first constraint satisfaction engine.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geo_kb import BBox


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════
# BBox geometry
# ═══════════════════════════════════════════════════════════════════════════

def intersect_bbox_lists(bbox_lists: list[list[BBox]]) -> list[BBox]:
    """N-way pairwise intersection of bbox lists.

    Sequential pairwise intersection using the same algorithm as
    element_fusion.intersect_bboxes(). If any step yields empty,
    returns the empty list immediately.
    """
    if not bbox_lists:
        return []
    active = list(bbox_lists[0])
    for next_bboxes in bbox_lists[1:]:
        result = []
        for a in active:
            for b in next_bboxes:
                lat_min = max(a.lat_min, b.lat_min)
                lat_max = min(a.lat_max, b.lat_max)
                lng_min = max(a.lng_min, b.lng_min)
                lng_max = min(a.lng_max, b.lng_max)
                if lat_min < lat_max and lng_min < lng_max:
                    result.append(BBox(
                        lat_min=lat_min, lat_max=lat_max,
                        lng_min=lng_min, lng_max=lng_max,
                        label=f"{a.label} & {b.label}",
                    ))
        if not result:
            return []
        active = result
    return active


def bbox_total_area_km2(bboxes: list[BBox]) -> float:
    """Sum of approximate areas of all bboxes in km^2."""
    return sum(b.area_km2 for b in bboxes)


def compute_constraint_power(
    bboxes_before: list[BBox],
    bboxes_after: list[BBox],
) -> float:
    """Area reduction ratio from before to after.

    Returns 0.0 = no reduction, 1.0 = eliminated everything.
    """
    area_before = bbox_total_area_km2(bboxes_before)
    area_after = bbox_total_area_km2(bboxes_after)
    if area_before <= 0:
        return 0.0
    return 1.0 - (area_after / area_before)


def compute_centroid_and_radius(
    bboxes: list[BBox],
) -> tuple[float, float, float]:
    """Area-weighted centroid + bounding radius of a bbox list.

    Returns (center_lat, center_lng, radius_km).
    Uses the same algorithm as element_fusion.compute_bbox_center_radius().
    """
    if not bboxes:
        return 35.0, 105.0, 2500.0

    if len(bboxes) == 1:
        b = bboxes[0]
        dlat = (b.lat_max - b.lat_min) * 111.0 / 2
        dlng = (b.lng_max - b.lng_min) * math.cos(math.radians(b.center_lat)) * 111.0 / 2
        return b.center_lat, b.center_lng, max(math.sqrt(dlat**2 + dlng**2), 10.0)

    total_lat = sum(b.center_lat * b.area_km2 for b in bboxes)
    total_lng = sum(b.center_lng * b.area_km2 for b in bboxes)
    total_area = sum(b.area_km2 for b in bboxes)
    center_lat = total_lat / total_area if total_area > 0 else 35.0
    center_lng = total_lng / total_area if total_area > 0 else 105.0

    max_dist = 0.0
    for b in bboxes:
        for lat, lng in [(b.lat_min, b.lng_min), (b.lat_min, b.lng_max),
                          (b.lat_max, b.lng_min), (b.lat_max, b.lng_max)]:
            dlat = math.radians(lat - center_lat)
            dlng = math.radians(lng - center_lng)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(center_lat)) * math.cos(math.radians(lat)) *
                 math.sin(dlng / 2) ** 2)
            d = 6371.0 * 2 * math.asin(math.sqrt(min(a, 1.0)))
            max_dist = max(max_dist, d)
    return center_lat, center_lng, max(max_dist, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# Explanation generation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConstraintStep:
    """Record of one constraint application step."""
    name: str                          # "elevation_4500m"
    source_type: str                   # "sensor_elevation"|"sensor_climate"|"element"|"compound"
    area_before_km2: float
    area_after_km2: float
    prune_ratio: float                 # fraction of area eliminated
    bbox_count_before: int = 0
    bbox_count_after: int = 0
    was_relaxed: bool = False
    was_dropped: bool = False
    metadata: dict = field(default_factory=dict)


def explain_step(step: ConstraintStep) -> str:
    """One-line human-readable explanation for a single constraint step."""
    if step.was_dropped:
        return (f"  [{step.source_type}] {step.name}: DROPPED "
                f"(would eliminate {step.prune_ratio:.0%} of remaining area)")
    relax = " (relaxed)" if step.was_relaxed else ""
    if step.prune_ratio < 0.01:
        return (f"  [{step.source_type}] {step.name}: "
                f"{step.area_after_km2:,.0f} km^2 "
                f"(minimal change{relax})")
    return (f"  [{step.source_type}] {step.name}: "
            f"{step.area_before_km2:,.0f} -&gt; {step.area_after_km2:,.0f} km^2 "
            f"| eliminated {step.prune_ratio:.0%}{relax}")


def explain_chain(
    steps: list[ConstraintStep],
    final_lat: float,
    final_lng: float,
    final_uncertainty_km: float,
    dropped: Optional[list[str]] = None,
) -> str:
    """Multi-line explanation string for full constraint chain."""
    lines = ["[CONSTRAINT-CHAIN] Sensor-first spatial reasoning:"]
    china_area = 9_600_000  # km^2

    for step in steps:
        lines.append(explain_step(step))

    if dropped:
        lines.append(f"  Dropped (conflict): {', '.join(dropped)}")

    total_prune = 1.0 - (steps[-1].area_after_km2 / china_area) if steps else 0.0
    lines.append(
        f"  RESULT: ({final_lat:.4f}, {final_lng:.4f}) "
        f"uncertainty={final_uncertainty_km:.0f}km | "
        f"total prune={total_prune:.1%} of China"
    )
    return "\n".join(lines)
