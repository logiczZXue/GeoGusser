"""Benchmark inference speed and memory usage.

Usage:
    python scripts/benchmark_inference.py --model-dir models/qwen2-vl-7b-ov-int4 --image test.jpg
"""

import argparse
import time

import psutil


def benchmark(model_dir: str, image_path: str, device: str = "GPU", n_runs: int = 3):
    """Run inference benchmark and report timing + memory."""
    import openvino as ov
    from PIL import Image

    core = ov.Core()
    print(f"Available devices: {core.available_devices}")

    # Load model components
    vit_path = f"{model_dir}/openvino_vision_encoder_model.xml"
    llm_path = f"{model_dir}/openvino_language_model.xml"

    if not model_dir.endswith(".xml"):
        # Try loading as a composite model
        print(f"Loading model from {model_dir} on {device}")

    print(f"\nBenchmark: {n_runs} runs on device={device}")
    print("-" * 60)

    timings = []
    for i in range(n_runs):
        mem_before = psutil.Process().memory_info().rss / 1024**3
        t0 = time.time()

        # Placeholder: actual inference call depends on the model pipeline
        # This will be filled in once the OpenVINO engine is validated
        print(f"  Run {i+1}: executing...", end=" ", flush=True)

        t1 = time.time()
        mem_after = psutil.Process().memory_info().rss / 1024**3
        elapsed = t1 - t0
        timings.append(elapsed)
        print(f"{elapsed:.1f}s  (memory: {mem_after - mem_before:+.2f} GB)")

    print("-" * 60)
    avg = sum(timings) / len(timings)
    print(f"Average: {avg:.1f}s  |  Min: {min(timings):.1f}s  |  Max: {max(timings):.1f}s")
    print(f"Peak memory: {max(psutil.Process().memory_info().rss / 1024**3 for _ in [0]):.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed and memory")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU", "NPU"])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    benchmark(args.model_dir, args.image, args.device, args.runs)


if __name__ == "__main__":
    main()
