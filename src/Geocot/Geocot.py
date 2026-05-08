"""GeoCoT: Geospatial Chain-of-Thought reasoning framework.

Implements a 3-stage coarse-to-fine pipeline for image geolocation:
  Stage 1 (Macro):  Continent-level — climate, vegetation, topography
  Stage 2 (Regional): Country-level — language, architecture, traffic, plates
  Stage 3 (Local):   City-level — sidewalk patterns, landmarks, synthesis

Based on the multi-turn CoT pattern from ImageGeoLocator_TestSet.py,
extended with structured prompts and local VLM support.
Tested with Qwen2-VL-2B-Instruct on RTX 4060 Laptop (8GB VRAM).
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GeoCoTStage(Enum):
    MACRO = "macro"
    REGIONAL = "regional"
    LOCAL = "local"


@dataclass
class GeoPrediction:
    city: str = "Unknown"
    country: str = "Unknown"
    continent: str = "Unknown"

    def __str__(self) -> str:
        return f"{self.city}, {self.country}, {self.continent}"


@dataclass
class GeoCoTResult:
    stage_outputs: dict = field(default_factory=dict)
    final_prediction: GeoPrediction = field(default_factory=GeoPrediction)
    reasoning_chain: str = ""

    @property
    def full_reasoning(self) -> str:
        parts = []
        for stage in GeoCoTStage:
            if stage.value in self.stage_outputs:
                parts.append(f"[{stage.value.upper()}]\n{self.stage_outputs[stage.value]}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_DEFAULT_PROMPTS = {
    GeoCoTStage.MACRO: (
        "Analyze this street view image for broad geographic context. Consider:\n"
        "- Climate zone (tropical, temperate, arid, polar) based on vegetation, sky, and soil\n"
        "- General topography (flat, mountainous, coastal, inland)\n"
        "- Vegetation type (palm trees, conifers, deciduous, sparse desert)\n"
        "- Overall urbanization level (dense city, suburban, rural)\n\n"
        "Provide a brief assessment of which continent(s) and broad region(s) this image could be from."
    ),
    GeoCoTStage.REGIONAL: (
        "Based on the macro-level assessment:\n\"{prev_output}\"\n\n"
        "Now examine this image for country-specific indicators:\n"
        "- Language on signs, buildings, or vehicles\n"
        "- Architectural style (roof shapes, wall colors, building materials)\n"
        "- Traffic direction and road infrastructure\n"
        "- License plate format and color\n"
        "- Utility pole design and fire hydrant style\n\n"
        "Which specific country (or small set of countries) does this image most likely come from?"
    ),
    GeoCoTStage.LOCAL: (
        "Based on the analysis so far:\n"
        "- Macro context: {macro_output}\n"
        "- Regional assessment: {regional_output}\n\n"
        "Now identify city-level details:\n"
        "- Sidewalk tile patterns (color, shape, arrangement)\n"
        "- Specific landmarks or recognizable buildings\n"
        "- Vehicle makes and models common to the region\n"
        "- Street naming conventions and signage style\n"
        "- Local business names or advertisements\n\n"
        "Provide your final geolocation as a coherent paragraph following this format:\n"
        '"The presence of [clue1] indicates [inference1]. [clue2] suggests [inference2]. ... '
        'This image was most likely taken in [city], [country], [continent]."'
    ),
}


class GeoCoTPrompt:
    """Manages stage-specific prompts and few-shot examples."""

    def __init__(self, prompts_dir: Optional[str] = None, few_shot_path: Optional[str] = None):
        self._prompts = {}
        self._load_prompts(prompts_dir)
        self._few_shots = self._load_few_shots(few_shot_path)

    def _load_prompts(self, prompts_dir: Optional[str]):
        for stage in GeoCoTStage:
            path = Path(prompts_dir) / f"{stage.value}.txt" if prompts_dir else None
            if path and path.exists():
                self._prompts[stage] = path.read_text(encoding="utf-8").strip()
            else:
                self._prompts[stage] = _DEFAULT_PROMPTS[stage]

    def _load_few_shots(self, path: Optional[str]) -> list[dict]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def get_prompt(self, stage: GeoCoTStage, prev_outputs: dict) -> str:
        template = self._prompts[stage]
        fmt = {}
        if "{prev_output}" in template and prev_outputs:
            last_key = list(prev_outputs.keys())[-1]
            fmt["prev_output"] = prev_outputs[last_key]
        if "{macro_output}" in template:
            fmt["macro_output"] = prev_outputs.get("macro", "")
        if "{regional_output}" in template:
            fmt["regional_output"] = prev_outputs.get("regional", "")
        try:
            return template.format(**fmt)
        except KeyError:
            return template

    def build_few_shot_prefix(self, continent_hint: str = "") -> str:
        if not self._few_shots:
            return ""
        examples = self._few_shots[:3]
        parts = ["Here are some examples of geographic reasoning:\n"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:\n{ex.get('output', '')}\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prediction extraction (offline, no GPT needed)
# ---------------------------------------------------------------------------

_COUNTRY_TO_CONTINENT = {
    "china": "Asia", "japan": "Asia", "thailand": "Asia", "south korea": "Asia",
    "india": "Asia", "vietnam": "Asia", "indonesia": "Asia", "malaysia": "Asia",
    "philippines": "Asia", "singapore": "Asia", "taiwan": "Asia",
    "france": "Europe", "germany": "Europe", "italy": "Europe", "spain": "Europe",
    "united kingdom": "Europe", "uk": "Europe", "netherlands": "Europe",
    "russia": "Europe", "portugal": "Europe", "poland": "Europe",
    "united states": "North America", "usa": "North America", "us": "North America",
    "america": "North America",
    "canada": "North America", "mexico": "North America",
    "brazil": "South America", "argentina": "South America",
    "colombia": "South America", "chile": "South America",
    "australia": "Oceania", "new zealand": "Oceania",
    "nigeria": "Africa", "south africa": "Africa", "egypt": "Africa", "kenya": "Africa",
}

_LANDMARK_TO_LOCATION = {
    "brooklyn bridge": ("New York City", "United States"),
    "statue of liberty": ("New York City", "United States"),
    "eiffel tower": ("Paris", "France"),
    "big ben": ("London", "United Kingdom"),
    "tower bridge": ("London", "United Kingdom"),
    "colosseum": ("Rome", "Italy"),
    "sagrada familia": ("Barcelona", "Spain"),
    "tokyo tower": ("Tokyo", "Japan"),
    "sydney opera house": ("Sydney", "Australia"),
    "golden gate bridge": ("San Francisco", "United States"),
    "notre dame": ("Paris", "France"),
    "brandenburg gate": ("Berlin", "Germany"),
}

_CITY_TO_COUNTRY = {
    "new york": "United States", "new york city": "United States",
    "los angeles": "United States", "chicago": "United States",
    "san francisco": "United States", "miami": "United States",
    "london": "United Kingdom", "paris": "France", "berlin": "Germany",
    "rome": "Italy", "madrid": "Spain", "barcelona": "Spain",
    "tokyo": "Japan", "osaka": "Japan",
    "beijing": "China", "shanghai": "China", "shenzhen": "China",
    "hong kong": "China", "guangzhou": "China", "fuzhou": "China",
    "bangkok": "Thailand", "seoul": "South Korea", "singapore": "Singapore",
    "sydney": "Australia", "melbourne": "Australia",
    "toronto": "Canada", "vancouver": "Canada",
    "moscow": "Russia", "istanbul": "Turkey", "cairo": "Egypt", "dubai": "UAE",
}


def _infer_continent(country: str) -> str:
    return _COUNTRY_TO_CONTINENT.get(country.lower(), "Unknown")


def extract_prediction(text: str) -> GeoPrediction:
    """Extract city, country, continent from reasoning text.

    Strategy: look for LOCATION: marker first, then fall back to other patterns.
    Only scan the conclusion (last sentence) for landmarks/cities to avoid
    matching locations mentioned during reasoning.
    """
    # 0. Check for explicit LOCATION: marker
    location_pattern = re.compile(
        r"LOCATION:\s*([^,]+?),\s*([^,]+?),\s*(Asia|Africa|Europe|North America|South America|Oceania)",
        re.IGNORECASE,
    )
    m = location_pattern.search(text)
    if m:
        return GeoPrediction(city=m.group(1).strip(), country=m.group(2).strip(), continent=m.group(3).strip())
    patterns = [
        # "taken in / located in city, country, continent"
        re.compile(
            r"(?:taken in|located in|most likely (?:taken|located|found) in)\s+"
            r"([^,]+?),\s*([^,]+?),\s*(?:the\s+)?(Asia|Africa|Europe|North America|South America|Oceania)",
            re.IGNORECASE,
        ),
        # "city, country, continent" standalone
        re.compile(
            r"([A-Z][a-zA-Z\s]+?),\s*([A-Z][a-zA-Z\s]+?),\s*"
            r"(Asia|Africa|Europe|North America|South America|Oceania)",
            re.IGNORECASE,
        ),
        # "from the Country, specifically City"
        re.compile(
            r"from\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?),\s*specifically\s+([A-Z][a-zA-Z\s]+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "in City, Country" (no continent)
        re.compile(
            r"(?:taken in|located in|found in|in)\s+([^,]+?),\s*([A-Z][a-zA-Z\s]+?)(?:\.|$)",
            re.IGNORECASE,
        ),
    ]
    # Get the last sentence as conclusion
    sentences = re.split(r'[.!?]\s*', text.strip())
    conclusion = sentences[-1] if sentences else text
    if len(conclusion) < 30 and len(sentences) > 1:
        conclusion = sentences[-2] + ". " + conclusion
    conclusion_lower = conclusion.lower()

    for pattern in patterns:
        m = pattern.search(conclusion)
        if m:
            groups = [g.strip() if g else "" for g in m.groups()]
            if len(groups) == 3 and groups[2].lower() in {"asia", "africa", "europe", "north america", "south america", "oceania"}:
                return GeoPrediction(city=groups[0], country=groups[1], continent=groups[2])
            elif len(groups) == 2:
                country = groups[1] if len(groups[1]) > len(groups[0]) else groups[0]
                city = groups[0] if len(groups[0]) <= len(groups[1]) else groups[1]
                return GeoPrediction(city=city, country=country, continent=_infer_continent(country))

    # 2. Scan for landmarks in the conclusion
    for landmark, (city, country) in _LANDMARK_TO_LOCATION.items():
        if landmark in conclusion_lower:
            return GeoPrediction(city=city, country=country, continent=_infer_continent(country))

    # 3. Scan for city names in the conclusion (longest match first)
    for city, country in sorted(_CITY_TO_COUNTRY.items(), key=lambda x: -len(x[0])):
        if city in conclusion_lower:
            return GeoPrediction(city=city.title(), country=country, continent=_infer_continent(country))

    # 4. Scan for country names in the conclusion
    for country, continent in _COUNTRY_TO_CONTINENT.items():
        if country in conclusion_lower:
            return GeoPrediction(city="Unknown", country=country.title(), continent=continent)

    # 5. Last resort: scan full text for landmarks only
    text_lower = text.lower()
    for landmark, (city, country) in _LANDMARK_TO_LOCATION.items():
        if landmark in text_lower:
            return GeoPrediction(city=city, country=country, continent=_infer_continent(country))

    return GeoPrediction()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

MAX_IMAGE_PIXELS = 504 * 28 * 28  # ~395K pixels, fits in 8GB VRAM


def resize_image_for_vlm(image: Image.Image, max_pixels: int = 500000) -> Image.Image:
    """Resize image if too large, to avoid CUDA OOM."""
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GeoCoTPipeline:
    """3-stage coarse-to-fine geolocation reasoning pipeline."""

    def __init__(self, model_fn, prompt_config: Optional[dict] = None):
        self._model_fn = model_fn
        self._prompt = GeoCoTPrompt(
            prompts_dir=prompt_config.get("prompts_dir") if prompt_config else None,
            few_shot_path=prompt_config.get("few_shot_path") if prompt_config else None,
        )

    def run(self, image: Image.Image) -> GeoCoTResult:
        """Run the full 3-stage GeoCoT pipeline on a single image."""
        image = resize_image_for_vlm(image)
        result = GeoCoTResult()

        for stage in GeoCoTStage:
            prompt_text = self._prompt.get_prompt(stage, result.stage_outputs)

            if stage == GeoCoTStage.LOCAL:
                few_shot_prefix = self._prompt.build_few_shot_prefix()
                if few_shot_prefix:
                    prompt_text = few_shot_prefix + "\n\n" + prompt_text

            output = self._model_fn(image, prompt_text, result.stage_outputs)
            result.stage_outputs[stage.value] = output.strip()

        result.reasoning_chain = result.full_reasoning
        result.final_prediction = extract_prediction(result.stage_outputs.get("local", ""))

        return result


# ---------------------------------------------------------------------------
# Qwen2-VL backend (tested, working)
# ---------------------------------------------------------------------------

def create_qwen2vl_model_fn(
    model,
    processor,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Create a model_fn for GeoCoTPipeline using Qwen2-VL.

    This is the backend that has been tested and verified working.
    """
    from qwen_vl_utils import process_vision_info

    def model_fn(image: Image.Image, prompt_text: str, prev_outputs: dict) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.empty_cache()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        result = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return result.strip()

    return model_fn


def load_qwen2vl(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    max_pixels: int = MAX_IMAGE_PIXELS,
):
    """Load Qwen2-VL model and processor, return (model, processor, model_fn)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_fn = create_qwen2vl_model_fn(model, processor)
    return model, processor, model_fn


# ---------------------------------------------------------------------------
# Legacy HuggingFace backend (for LLaMA-3.2-Vision, untested on this GPU)
# ---------------------------------------------------------------------------

def create_hf_model_fn(model, processor, device, temperature=0.7, top_p=0.9, max_new_tokens=2048):
    """Create a model_fn for MllamaForConditionalGeneration (LLaMA-3.2-Vision)."""
    def model_fn(image: Image.Image, prompt_text: str, prev_outputs: dict) -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        for stage_key, stage_output in prev_outputs.items():
            conversation.append({"role": "assistant", "content": stage_output})

        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
            )

        decoded = processor.decode(output[0])
        response = decoded[len(prompt):]
        for tok in ["<|eot_id|>", "<|im_end|>", "<|end_of_text|>"]:
            response = response.replace(tok, "")
        return response.strip()

    return model_fn
