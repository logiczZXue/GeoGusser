"""Download VLM model weights for the TuXun project."""

import argparse
import os

from huggingface_hub import snapshot_download

SUPPORTED_MODELS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


def main():
    parser = argparse.ArgumentParser(description="Download VLM model weights")
    parser.add_argument(
        "model",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: D:/Geocomp/models/<model-name>)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HUGGINGFACE_TOKEN env var)",
    )
    args = parser.parse_args()

    model_id = SUPPORTED_MODELS[args.model]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", args.model
    )

    token = args.token or os.getenv("HUGGINGFACE_TOKEN")
    if not token and "llama" in args.model:
        print("LLaMA models require a HuggingFace token with Meta access approval.")
        print("Set HUGGINGFACE_TOKEN env var or pass --token")
        return

    print(f"Downloading {model_id} to {output_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        token=token,
    )
    print(f"Done. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
