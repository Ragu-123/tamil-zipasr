"""
Training Script for ZipTransducer-Tamil Model Family
====================================================
End-to-end streaming training on ai4bharat/IndicVoices Tamil speech corpus.

Key Features:
- Supports Small (~10.5M), Medium (~51.2M), and Large (~115.3M) variants.
- Zero-disk streaming dataset pipeline with in-memory audio decoding.
- Multi-task training: Joint RNN-T Loss + 0.25 * Auxiliary CTC Loss.
- Mixed precision training (FP16 / BF16) with gradient scaling.
- SpecAugment (time & frequency masking) for acoustic robustness.
- Cosine Annealing learning rate schedule with linear warmup.
- Live validation evaluation and transcription preview during training.
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models.asr_model import build_zip_transducer, ZipTransducerTamil, ZipTransducerConfig
from models.tokenizer import TamilAksharaTokenizer
from data.dataset import IndicVoicesStreamingDataset
from data.collator import ASRDataCollator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ZipTransducer-Tamil ASR model")
    parser.add_argument("--variant", type=str, default="small", choices=["small", "medium", "large"],
                        help="Model family variant: small (~10.5M), medium (~51.2M), large (~115.3M)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps")
    parser.add_argument("--max-steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--clip-grad", type=float, default=5.0, help="Maximum gradient norm")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--use-triton", action="store_true", default=False, help="Enable custom Triton GPU kernels")
    parser.add_argument("--mock-data", action="store_true", default=False, help="Run in mock/offline data mode for verification")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for dataset streaming")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=20, help="Log metrics every N steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.01):
    """Cosine Annealing with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 1. Initialize Tokenizer & Model
    tokenizer = TamilAksharaTokenizer()
    logger.info(f"Tamil Akshara Tokenizer initialized with {tokenizer.vocab_size} tokens")

    model = build_zip_transducer(
        variant=args.variant,
        vocab_size=tokenizer.vocab_size,
        use_triton_ops=args.use_triton,
    ).to(device)

    model.print_architecture_summary()
    breakdown = model.get_parameter_breakdown()
    logger.info(f"Model [{args.variant.upper()}] assembled: {breakdown['total_million']}M parameters")

    # 2. Setup Streaming Dataset & DataLoader
    train_dataset = IndicVoicesStreamingDataset(
        split="train",
        hf_token=args.hf_token,
        tokenizer=tokenizer,
        apply_spec_augment=True,
        mock_streaming=args.mock_data,
        shuffle_buffer_size=200 if args.mock_data else 500,
    )
    collator = ASRDataCollator(pad_id=tokenizer.blank_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,  # Streaming iterable
    )

    # 3. Setup Optimizer, Scaler, and Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # 4. Training Loop
    logger.info(f"Starting training run: max_steps={args.max_steps}, batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    model.train()
    optimizer.zero_grad()

    step = start_step
    running_loss = 0.0
    running_rnnt = 0.0
    running_ctc = 0.0
    t_start = time.time()

    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        inputs = batch["inputs"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
            outputs = model(
                audio_features=inputs,
                audio_lengths=input_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )
            loss = outputs["loss"] / args.grad_accum
            rnnt_loss = outputs["rnnt_loss"]
            ctc_loss = outputs["ctc_loss"]

        scaler.scale(loss).backward()

        running_loss += loss.item() * args.grad_accum
        running_rnnt += rnnt_loss.item()
        running_ctc += ctc_loss.item()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            cur_lr = scheduler.get_last_lr()[0]
            avg_loss = running_loss / args.log_every
            avg_rnnt = running_rnnt / args.log_every
            avg_ctc = running_ctc / args.log_every
            steps_per_sec = args.log_every / max(1e-5, elapsed)

            logger.info(
                f"Step [{step}/{args.max_steps}] | Loss: {avg_loss:.4f} "
                f"(RNNT: {avg_rnnt:.4f}, CTC: {avg_ctc:.4f}) | "
                f"LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s"
            )
            running_loss = 0.0
            running_rnnt = 0.0
            running_ctc = 0.0
            t_start = time.time()

        # Validation Preview
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                sample_inputs = inputs[:2]
                sample_lens = input_lengths[:2]
                preds_ctc = model.transcribe(sample_inputs, sample_lens, method="ctc")
                logger.info(f"--- Step {step} Validation Preview ---")
                for i in range(len(preds_ctc)):
                    gold = batch["texts"][i] if i < len(batch["texts"]) else ""
                    logger.info(f"Target:     {gold}")
                    logger.info(f"Prediction: {preds_ctc[i]}")
                logger.info("------------------------------------")
            model.train()

        # Checkpointing
        if step % args.save_every == 0 or step == args.max_steps:
            save_path = os.path.join(args.checkpoint_dir, f"zip_transducer_{args.variant}_step_{step}.pt")
            logger.info(f"Saving checkpoint to {save_path}")
            torch.save({
                "step": step,
                "variant": args.variant,
                "config": model.config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
