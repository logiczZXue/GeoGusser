"""Flask demo for TuXun: upload a street view image and get geolocation with CoT reasoning."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, render_template, request, jsonify
from src.Geocot.Geocot import GeoCoTPipeline, GeoCoTResult
from src.Geocot.prediction_extractor import format_prediction

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

pipeline: GeoCoTPipeline | None = None


def init_pipeline(model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Initialize the GeoCoT pipeline with a VLM backend."""
    global pipeline
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from src.Geocot.Geocot import create_hf_model_fn

    print(f"Loading model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model_fn = create_hf_model_fn(model, processor, "cuda" if torch.cuda.is_available() else "cpu")

    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "Geocot", "prompts")
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": prompts_dir})
    print("Pipeline ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Run GeoCoT on an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    from PIL import Image
    import io

    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    result = pipeline.run(image)

    return jsonify({
        "stages": {
            "macro": result.stage_outputs.get("macro", ""),
            "regional": result.stage_outputs.get("regional", ""),
            "local": result.stage_outputs.get("local", ""),
        },
        "prediction": format_prediction(result.final_prediction),
        "full_reasoning": result.reasoning_chain,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TuXun Flask demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    init_pipeline(args.model)
    app.run(host=args.host, port=args.port, debug=False)
