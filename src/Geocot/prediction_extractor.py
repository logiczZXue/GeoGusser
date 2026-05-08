"""Offline prediction extractor — extracts city, country, continent from reasoning text.

Replaces the GPT-4o-based ask_gpt() post-processing from inference_example.py.
"""

from .Geocot import GeoPrediction, extract_prediction

__all__ = ["extract_prediction", "GeoPrediction", "format_prediction"]


def format_prediction(prediction: GeoPrediction) -> str:
    """Format as 'city, country, continent' string (matching existing evaluation format)."""
    return f"{prediction.city}, {prediction.country}, {prediction.continent}"
