"""Offline lat/lng predictor — replaces GPT-4o + Google Geocoding API.

Uses a pre-built city/country -> coordinates lookup table from the tuxun dataset.
"""

import json
import os
import re
from typing import Optional

from geopy.geocoders import Nominatim


class OfflineLatLngPredictor:
    """Convert city/country/country text to lat/lng coordinates offline."""

    def __init__(self, geocoding_cache_path: str = None):
        self._cache = {}
        if geocoding_cache_path and os.path.exists(geocoding_cache_path):
            with open(geocoding_cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def predict(self, city: str, country: str) -> tuple[float, float]:
        """Look up coordinates from cache. Returns (lat, lng) or (0.0, 0.0)."""
        # Try exact match first
        key = f"{city.lower()}, {country.lower()}"
        if key in self._cache:
            entry = self._cache[key]
            return entry["lat"], entry["lng"]

        # Try country-only match
        country_key = country.lower()
        for k, v in self._cache.items():
            if country_key in k:
                return v["lat"], v["lng"]

        return 0.0, 0.0

    def predict_from_text(self, text: str) -> tuple[float, float]:
        """Extract city/country from text and predict coordinates."""
        # Pattern: "city, country, continent"
        m = re.search(
            r"([A-Za-z\s]+?),\s*([A-Za-z\s]+?),\s*(?:Asia|Africa|Europe|North America|South America|Oceania)",
            text,
        )
        if m:
            return self.predict(m.group(1).strip(), m.group(2).strip())
        return 0.0, 0.0


def build_geocoding_cache(dataset_path: str, output_path: str, max_per_city: int = 5):
    """Build a geocoding cache from the tuxun dataset.

    Creates a JSON file mapping "city, country" -> {lat, lng}.
    """
    import csv

    csv.field_size_limit(100000000)
    cache = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data = json.loads(row.get("data", "{}"))
                rounds_data = data.get("rounds", [])
            except (json.JSONDecodeError, KeyError):
                continue

            for rd in rounds_data:
                if not isinstance(rd, dict):
                    continue
                lat = rd.get("lat")
                lng = rd.get("lng")
                nation = rd.get("nation", "")
                if lat and lng and nation:
                    key = nation.lower()
                    if key not in cache:
                        cache[key] = {"lat": float(lat), "lng": float(lng)}
                    elif isinstance(cache[key], list):
                        if len(cache[key]) < max_per_city:
                            cache[key].append({"lat": float(lat), "lng": float(lng)})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Geocoding cache saved to {output_path} ({len(cache)} entries)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Offline lat/lng prediction")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build geocoding cache from dataset")
    build_parser.add_argument("--dataset", type=str, required=True)
    build_parser.add_argument("--output", type=str, default="data/geocoding_cache.json")

    predict_parser = subparsers.add_parser("predict", help="Predict coordinates")
    predict_parser.add_argument("--cache", type=str, default="data/geocoding_cache.json")
    predict_parser.add_argument("--city", type=str, required=True)
    predict_parser.add_argument("--country", type=str, required=True)

    args = parser.parse_args()

    if args.command == "build":
        build_geocoding_cache(args.dataset, args.output)
    elif args.command == "predict":
        predictor = OfflineLatLngPredictor(args.cache)
        lat, lng = predictor.predict(args.city, args.country)
        print(f"{args.city}, {args.country}: lat={lat}, lng={lng}")
