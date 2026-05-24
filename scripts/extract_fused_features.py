"""Pre-extract visual features + VLM hidden states for FusedMoE training.

For each training image, extracts:
  - visual_features: (1536,) from Qwen2-VL vision encoder
  - vlm_hidden_macro/regional/local/fused: (1536,) each, from VLM forward pass
  - coords_norm: (2,) normalized lat/lng
  - scene_type: int (0-3) from precomputed labels

Output: output/regression/fused_features.pt — a dict with:
  "visual": (N, 1536)
  "vlm_macro/regional/local/fused": (N, 1536) each
  "coords": (N, 2) normalized
  "scene_types": (N,) int
  "paths": list[str] of image file paths

Usage:
    python scripts/extract_fused_features.py
    python scripts/extract_fused_features.py --max-images 100  # quick test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from Geocot.Geocot import GeoCoTStage, GeoCoTResult  # noqa: E402


def load_prompts(enable_knowledge: bool = True) -> dict:
    """Load GeoCoT stage prompts (enhanced versions if available)."""
    from Geocot.Geocot import GeoCoTPrompt

    prompts_dir = _src / "Geocot" / "prompts"
    if not (prompts_dir / "macro.txt").exists():
        prompts_dir = None

    prompt_mgr = GeoCoTPrompt(
        prompts_dir=str(prompts_dir) if prompts_dir else None,
        few_shot_path=None,
        enable_knowledge_injection=enable_knowledge,
    )
    return prompt_mgr


def load_scene_labels(labels_file: str) -> dict[int, int]:
    """Load scene type labels: {sample_idx: scene_type}."""
    with open(labels_file, "r") as f:
        labels = json.load(f)
    # The JSON maps image paths to scene type strings/ints
    out = {}
    for i, (path, st) in enumerate(labels.items()):
        out[i] = int(st) if isinstance(st, (int, str)) else 0
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract visual + VLM hidden features for FusedMoE training"
    )
    parser.add_argument(
        "--data", type=str, default="data/mapillary",
        help="Image data root directory"
    )
    parser.add_argument(
        "--index", type=str, default="data/merged_geotagged.json",
        help="JSON mapping image paths → (lat, lng)"
    )
    parser.add_argument(
        "--labels", type=str, default="output/regression/scene_labels.json",
        help="Scene type labels JSON"
    )
    parser.add_argument(
        "--output", type=str, default="output/regression/fused_features.pt",
        help="Output file path"
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to process (0 = all)"
    )
    parser.add_argument(
        "--save-interval", type=int, default=200,
        help="Save incremental checkpoint every N images"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip images already in an existing output file"
    )
    args = parser.parse_args()

    # ── Load index ─────────────────────────────────────────────────────
    with open(args.index, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    all_paths = list(geo_data.keys())
    if args.max_images > 0:
        all_paths = all_paths[:args.max_images]

    # Filter to existing images
    valid_paths = []
    valid_coords = []
    for p in all_paths:
        abs_path = p
        if not os.path.isabs(p):
            abs_path = str(Path(p).resolve())
        if os.path.exists(abs_path):
            valid_paths.append(abs_path)
            valid_coords.append(geo_data[p])
        else:
            alt = Path(args.data).resolve().parent / p
            if os.path.exists(alt):
                valid_paths.append(str(alt))
                valid_coords.append(geo_data[p])

    print(f"Images: {len(valid_paths)} valid out of {len(all_paths)}")

    # ── Load scene labels ──────────────────────────────────────────────
    scene_types = {}
    if os.path.exists(args.labels):
        with open(args.labels, "r") as f:
            raw_labels = json.load(f)
        # Map image paths → scene type ints
        for path_str, st in raw_labels.items():
            scene_types[path_str] = int(st) if isinstance(st, (int, str)) else 0
        print(f"Scene labels loaded: {len(scene_types)} entries")

    # ── Load VLM ───────────────────────────────────────────────────────
    print("Loading VLM...")
    from Geocot.Geocot import load_qwen2vl
    from regression.feature_extractor import VisualFeatureExtractor
    from regression.vlm_hidden_extractor import VLMHiddenExtractor

    model, processor, _ = load_qwen2vl("Qwen/Qwen2-VL-2B-Instruct")
    vis_extractor = VisualFeatureExtractor(model, processor)
    hidden_extractor = VLMHiddenExtractor(model, processor)

    # ── Load prompts ───────────────────────────────────────────────────
    prompt_mgr = load_prompts(enable_knowledge=True)

    # Pre-resolve macro prompt (doesn't depend on prev_outputs)
    macro_prompt = prompt_mgr.get_prompt(
        GeoCoTStage.MACRO, prev_outputs={}
    )

    # We'll build regional/local on the fly with placeholder prev outputs
    dummy_prev = {"macro": "placeholder", "regional": "placeholder"}
    regional_prompt_base = prompt_mgr.get_prompt(
        GeoCoTStage.REGIONAL, prev_outputs=dummy_prev
    )
    local_prompt_base = prompt_mgr.get_prompt(
        GeoCoTStage.LOCAL, prev_outputs=dummy_prev
    )

    print(f"Macro prompt  : {len(macro_prompt)} chars")
    print(f"Regional prompt: {len(regional_prompt_base)} chars")
    print(f"Local prompt   : {len(local_prompt_base)} chars")

    # ── Extract features ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = {
        "visual": [],
        "vlm_macro": [],
        "vlm_regional": [],
        "vlm_local": [],
        "vlm_fused": [],
        "coords": [],
        "scene_types": [],
        "paths": [],
    }

    def save_checkpoint(suffix=""):
        path = args.output
        if suffix:
            path = path.replace(".pt", f"{suffix}.pt")
        torch.save(results, path)
        print(f"  Saved {len(results['paths'])} records to {path}")

    for i, (img_path, (true_lat, true_lng)) in enumerate(
        tqdm(list(zip(valid_paths, valid_coords)), desc="Extracting", unit="img")
    ):
        try:
            image = Image.open(img_path).convert("RGB")

            # 1. Visual features
            visual_feat = vis_extractor.extract(image).cpu()

            # 2. VLM hidden states (3 stages)
            vlm_hidden = hidden_extractor.extract_3stage(
                image,
                macro_prompt=macro_prompt,
                regional_prompt=regional_prompt_base,
                local_prompt=local_prompt_base,
            )

            # 3. Normalize coords
            lat_norm = true_lat / 90.0
            lng_norm = true_lng / 180.0

            # 4. Scene type
            scene = scene_types.get(img_path, 3)

            results["visual"].append(visual_feat)
            results["vlm_macro"].append(vlm_hidden["macro"].cpu())
            results["vlm_regional"].append(vlm_hidden["regional"].cpu())
            results["vlm_local"].append(vlm_hidden["local"].cpu())
            results["vlm_fused"].append(vlm_hidden["fused"].cpu())
            results["coords"].append(torch.tensor([lat_norm, lng_norm]))
            results["scene_types"].append(scene)
            results["paths"].append(img_path)

        except Exception as e:
            tqdm.write(f"  Error [{img_path}]: {e}")
            continue

        # Incremental save
        if (i + 1) % args.save_interval == 0:
            save_checkpoint(f"_ckpt_{i+1}")

    # ── Final save ─────────────────────────────────────────────────────
    # Stack lists into tensors
    results["visual"] = torch.stack(results["visual"])
    results["vlm_macro"] = torch.stack(results["vlm_macro"])
    results["vlm_regional"] = torch.stack(results["vlm_regional"])
    results["vlm_local"] = torch.stack(results["vlm_local"])
    results["vlm_fused"] = torch.stack(results["vlm_fused"])
    results["coords"] = torch.stack(results["coords"])
    results["scene_types"] = torch.tensor(results["scene_types"], dtype=torch.long)

    torch.save(results, args.output)
    print(f"\nFinal: {len(results['paths'])} records saved to {args.output}")
    print(f"  visual:      {results['visual'].shape}")
    print(f"  vlm_macro:   {results['vlm_macro'].shape}")
    print(f"  vlm_regional:{results['vlm_regional'].shape}")
    print(f"  vlm_local:   {results['vlm_local'].shape}")
    print(f"  vlm_fused:   {results['vlm_fused'].shape}")
    print(f"  coords:      {results['coords'].shape}")
    print(f"  scene_types: {results['scene_types'].shape}")


if __name__ == "__main__":
    main()
