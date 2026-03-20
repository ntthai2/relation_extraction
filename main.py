"""Main entry point for relation extraction experiments."""

import argparse
import os
from contextlib import contextmanager
import sys

from config import (
    DEFAULT_OUTPUT_DIR, DEFAULT_LOG_PATH, DEFAULT_RUN_NAME,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO,
    PATIENCE, MIN_DELTA, SEED, MODEL_VARIANTS, LOSS_VARIANTS
)
from run_exp import run_experiment_suite


class TeeStream:
    """Tee output to both stdout and a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


@contextmanager
def tee_output(log_path):
    """Context manager to tee stdout/stderr to file."""
    if not log_path:
        yield
        return

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run relation extraction experiments with different model/loss configs."
    )
    
    # Experiment configuration
    parser.add_argument(
        "--model-type",
        choices=["scibert", "deberta", "roberta_large", "bert_large", "spert", "plmarker", "pure_lite"],
        default="scibert",
        help="Model type to use"
    )
    parser.add_argument(
        "--model-variants",
        default="e1e2_concat",
        help="Comma-separated model pooling variants: e1e2_concat, cls_only, mean_pool, e1_only"
    )
    parser.add_argument(
        "--loss-variants",
        default="weighted_ce",
        help="Comma-separated loss variants: weighted_ce, ce_uniform, focal, label_smooth"
    )
    parser.add_argument(
        "--frozen-bert",
        action="store_true",
        help="Freeze BERT parameters (only train classifier head)"
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds for multi-seed runs (e.g., '13,21,42,87,100')"
    )
    
    # Advanced training tricks
    parser.add_argument(
        "--separate-lr",
        action="store_true",
        help="Use separate learning rates for encoder and classifier head"
    )
    parser.add_argument(
        "--llrd",
        action="store_true",
        help="Use layer-wise learning rate decay (only if --separate-lr insufficient)"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default 1, no accumulation)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable symmetric relation augmentation (CONJUNCTION, COMPARE)"
    )
    parser.add_argument(
        "--undersample-conjunction",
        action="store_true",
        help="Undersample CONJUNCTION to reduce dominance"
    )
    parser.add_argument(
        "--undersample-target",
        type=int,
        default=250,
        help="Target count for undersampled label"
    )
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO, help="Warmup ratio")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA, help="Min delta for patience")
    
    # Output
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory root")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Experiment name")
    parser.add_argument("--log-file", default=DEFAULT_LOG_PATH, help="Log file path (default: auto-save to runs/<run_name>/train.log)")
    
    # Debug
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug: cap samples per split; 0 means use full data"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny debug experiment (1 epoch, 128 samples)"
    )
    
    return parser.parse_args()



def main():
    args = parse_args()

    if args.separate_lr and args.llrd:
        argparse.ArgumentParser().error("--separate-lr and --llrd are mutually exclusive.")
    
    # Determine log path: use explicit --log-file if provided, else auto-generate
    if args.log_file.strip():
        log_path = args.log_file.strip()
    else:
        # Auto-generate: runs/<run_name>/train.log
        log_path = os.path.join(args.output_dir, args.run_name, "train.log")

    with tee_output(log_path):
        # Parse model variants
        model_vars = [m.strip() for m in args.model_variants.split(",") if m.strip()]
        for mv in model_vars:
            if mv not in MODEL_VARIANTS:
                raise ValueError(f"Unknown model variant: {mv}")

        # Parse loss variants
        loss_vars = [l.strip() for l in args.loss_variants.split(",") if l.strip()]
        for lv in loss_vars:
            if lv not in LOSS_VARIANTS:
                raise ValueError(f"Unknown loss variant: {lv}")

        # Parse seeds
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

        # Smoke test mode
        if args.smoke_test:
            args.epochs = 1
            args.patience = 1
            args.max_samples = 128
            args.run_name = f"{args.run_name}_smoke"

        # Run experiment suite
        run_experiment_suite(
            model_variants=model_vars,
            loss_variants=loss_vars,
            frozen_bert=args.frozen_bert,
            seeds=seeds,
            model_type=args.model_type,
            separate_lr=args.separate_lr,
            llrd=args.llrd,
            grad_accum_steps=args.grad_accum_steps,
            augment=args.augment,
            undersample_conjunction=args.undersample_conjunction,
            undersample_target=args.undersample_target,
            max_len=args.max_len,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            patience=args.patience,
            min_delta=args.min_delta,
            output_dir=args.output_dir,
            run_name=args.run_name,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
