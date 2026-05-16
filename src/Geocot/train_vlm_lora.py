"""LoRA fine-tune Qwen2-VL-2B for fine-grained China geo description.

Trains the VLM to produce detailed geological/botanical/cultural descriptions
from images, using KB-augmented instruction-response pairs.

Dependencies: peft, bitsandbytes (optional for 4-bit)
Usage:
    python -m src.Geocot.train_vlm_lora \
        --data data/vlm_finetune/train.jsonl \
        --output output/vlm_lora \
        --epochs 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ── Dataset ─────────────────────────────────────────────────────────────

class VLMFineTuneDataset(Dataset):
    """Load (image, instruction, response) triples for VLM fine-tuning."""

    def __init__(self, jsonl_path: str, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
        self.records = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if os.path.exists(rec["image"]):
                    self.records.append(rec)

        if not self.records:
            raise ValueError(f"No valid records found in {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image
        image = Image.open(rec["image"]).convert("RGB")

        # Build conversation with image
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": rec["instruction"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": rec["response"]},
            ]},
        ]

        # Apply chat template + process vision
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Squeeze batch dim
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)

        # Create labels from input_ids (causal LM objective)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


def collate_vlm(batch):
    """Pad and collate VLM inputs."""
    from torch.nn.utils.rnn import pad_sequence

    pad_keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
    collated = {}
    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if key in pad_keys:
            collated[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key in ("image_grid_thw",):
            collated[key] = torch.stack(tensors)
        elif key in ("pixel_values",):
            # pixel_values have different lengths — use cat with batch dim
            # Qwen2-VL handles variable-length pixel_values
            collated[key] = torch.cat(tensors, dim=0)
        else:
            collated[key] = tensors[0] if isinstance(tensors[0], torch.Tensor) else tensors

    # Mask padding tokens in labels
    if "labels" in collated and "attention_mask" in collated:
        collated["labels"][collated["attention_mask"] == 0] = -100

    return collated


# ── Training ────────────────────────────────────────────────────────────

def train_lora(
    data_file: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    output_dir: str = "output/vlm_lora",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    load_in_4bit: bool = False,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 100,
    logging_steps: int = 10,
    max_length: int = 512,
):
    """Main LoRA training function.

    Args:
        data_file: JSONL file with image/instruction/response fields
        model_name: HuggingFace model ID
        output_dir: where to save adapter weights
        epochs: number of training epochs
        batch_size: per-device batch size (keep small for VL)
        lr: learning rate
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha
        load_in_4bit: use 4-bit quantization
        gradient_accumulation_steps: effective batch = batch_size × accum
        save_steps: checkpoint every N steps
        logging_steps: log every N steps
        max_length: max token length for text
    """
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading {model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        quantization_config=bnb_config,
        device_map="auto" if load_in_4bit else None,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

    # Reduce image resolution to limit visual tokens per image
    # Default: shortest_edge=3136, longest_edge=12845056 → up to ~50k tokens
    # We set max_pixels to ~500k → ~2500 visual tokens (fits in 8GB VRAM)
    processor.image_processor.size["shortest_edge"] = 448
    processor.image_processor.size["longest_edge"] = 401344  # ~448*896

    if not load_in_4bit:
        model.to(device)

    # ── LoRA config ─────────────────────────────────────────────────
    # Target Qwen2-VL attention projection layers
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    model.train()

    # ── Data ────────────────────────────────────────────────────────
    dataset = VLMFineTuneDataset(data_file, processor, max_length=max_length)
    print(f"Dataset: {len(dataset)} records")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_vlm,
    )

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = (len(dataloader) * epochs) // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    global_step = 0
    accum_loss = 0.0

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / gradient_accumulation_steps
            loss.backward()

            accum_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    pbar.set_postfix({
                        "loss": f"{accum_loss * gradient_accumulation_steps:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })
                    accum_loss = 0.0

                if global_step % save_steps == 0:
                    checkpoint = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint)
                    print(f"\nSaved checkpoint to {checkpoint}")

    # ── Save final ──────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(final_path)
    print(f"\nLoRA adapter saved to {final_path}")

    # Merge and save full model (optional, may need more VRAM)
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(output_dir, "merged"))
        processor.save_pretrained(os.path.join(output_dir, "merged"))
        print(f"Merged model saved to {output_dir}/merged")
    except Exception as e:
        print(f"Merging skipped: {e}")

    return model


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2-VL for geo description")
    parser.add_argument("--data", type=str, default="data/vlm_finetune/train.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output", type=str, default="output/vlm_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--4bit", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train_lora(
        data_file=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        load_in_4bit=args.__dict__.get("4bit", False),
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
