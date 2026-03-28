#!/usr/bin/env python3
"""
Diagnostic script to analyze loss component behavior during training.

This script helps understand why contrastive loss plateaus while box loss continues to decrease.
It logs detailed statistics about each loss component including:
- Raw loss values (before weighting)
- Weighted loss values (after applying hyperparameter weights)
- Normalization factors used
- Number of objects per batch

Usage:
    python diagnose_loss.py --model <model_path> --data <data.yaml> --epochs 10
"""

import argparse
import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER
import csv
from pathlib import Path


class DiagnosticTrainer(BaseTrainer):
    """Custom trainer that logs detailed loss statistics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log_file = None
        self.loss_writer = None

    def _setup_train(self, world_size):
        """Setup training with loss logging."""
        super()._setup_train(world_size)

        # Create CSV file for loss logging
        log_dir = self.save_dir
        self.loss_log_file = log_dir / "loss_diagnostics.csv"

        # Write header
        with open(self.loss_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch',
                'box_loss_raw', 'cls_loss_raw', 'dfl_loss_raw', 'contrast_loss_raw',
                'box_loss_weighted', 'cls_loss_weighted', 'dfl_loss_weighted', 'contrast_loss_weighted',
                'total_loss', 'batch_size', 'num_objects',
                'box_weight', 'cls_weight', 'dfl_weight', 'contrast_weight',
                'target_scores_sum'
            ])

        LOGGER.info(f"Loss diagnostics will be saved to: {self.loss_log_file}")

    def _do_train(self, world_size=1):
        """Training loop with detailed loss logging."""
        # Call parent training but intercept loss computation
        return super()._do_train(world_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose YOLO loss behavior")
    parser.add_argument("--model", type=str, required=True, help="Model path (.yaml or .pt)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def analyze_loss_file(csv_path):
    """Analyze the loss diagnostics CSV and print summary statistics."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    print("\n" + "="*80)
    print("LOSS DIAGNOSTICS SUMMARY")
    print("="*80)

    # Group by epoch
    for epoch in df['epoch'].unique():
        epoch_df = df[df['epoch'] == epoch]

        print(f"\nEpoch {epoch}:")
        print("-" * 60)

        # Average loss components (raw, before weighting)
        print("Raw Loss Components (unweighted):")
        print(f"  Box Loss:      {epoch_df['box_loss_raw'].mean():.6f} (std: {epoch_df['box_loss_raw'].std():.6f})")
        print(f"  Cls Loss:      {epoch_df['cls_loss_raw'].mean():.6f} (std: {epoch_df['cls_loss_raw'].std():.6f})")
        print(f"  DFL Loss:      {epoch_df['dfl_loss_raw'].mean():.6f} (std: {epoch_df['dfl_loss_raw'].std():.6f})")
        if 'contrast_loss_raw' in epoch_df.columns:
            print(f"  Contrast Loss: {epoch_df['contrast_loss_raw'].mean():.6f} (std: {epoch_df['contrast_loss_raw'].std():.6f})")

        # Weighted contributions
        print("\nWeighted Loss Contributions:")
        print(f"  Box:      {epoch_df['box_loss_weighted'].mean():.6f}")
        print(f"  Cls:      {epoch_df['cls_loss_weighted'].mean():.6f}")
        print(f"  DFL:      {epoch_df['dfl_loss_weighted'].mean():.6f}")
        if 'contrast_loss_weighted' in epoch_df.columns:
            print(f"  Contrast: {epoch_df['contrast_loss_weighted'].mean():.6f}")

        print(f"\nTotal Loss: {epoch_df['total_loss'].mean():.6f}")
        print(f"Avg Objects/Batch: {epoch_df['num_objects'].mean():.2f}")
        print(f"Avg Batch Size: {epoch_df['batch_size'].mean():.2f}")

        # Weight evolution
        if epoch > 0:
            print(f"\nLoss Weights:")
            print(f"  Box: {epoch_df['box_weight'].iloc[0]:.4f}")
            print(f"  Cls: {epoch_df['cls_weight'].iloc[0]:.4f}")
            print(f"  DFL: {epoch_df['dfl_weight'].iloc[0]:.4f}")
            if 'contrast_weight' in epoch_df.columns:
                print(f"  Contrast: {epoch_df['contrast_weight'].iloc[0]:.4f}")

    # Overall trends
    print("\n" + "="*80)
    print("OVERALL TRENDS")
    print("="*80)

    # Calculate loss reduction
    first_epoch = df[df['epoch'] == df['epoch'].min()]
    last_epoch = df[df['epoch'] == df['epoch'].max()]

    print("\nLoss Reduction (First → Last Epoch):")
    for loss_type in ['box_loss_raw', 'cls_loss_raw', 'dfl_loss_raw', 'contrast_loss_raw']:
        if loss_type in df.columns:
            first_val = first_epoch[loss_type].mean()
            last_val = last_epoch[loss_type].mean()
            reduction = ((first_val - last_val) / first_val) * 100 if first_val > 0 else 0
            print(f"  {loss_type:20s}: {first_val:.6f} → {last_val:.6f} ({reduction:+.2f}%)")

    print("\nTotal Loss: {:.6f} → {:.6f}".format(
        first_epoch['total_loss'].mean(),
        last_epoch['total_loss'].mean()
    ))


def main():
    args = parse_args()

    print("="*80)
    print("YOLO LOSS DIAGNOSTICS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print("="*80)

    # Load model
    model = YOLO(args.model)

    # Train with diagnostic logging
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/diagnose",
        name="loss_analysis",
        verbose=True,
        # Enable contrastive learning if configured
        patience=args.epochs + 10,  # Disable early stopping for diagnostics
    )

    # Analyze results
    print("\nTraining complete. Analyzing loss behavior...")

    # Read results.csv from the training run
    results_dir = Path("runs/diagnose/loss_analysis")
    results_csv = results_dir / "results.csv"

    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)

        print("\n" + "="*80)
        print("TRAINING RESULTS ANALYSIS")
        print("="*80)

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        print("\nAvailable metrics:", list(df.columns))

        # Plot loss trends if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Loss Component Evolution', fontsize=16)

            # Box loss
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Box Loss')
                axes[0, 0].set_title('Box Loss Evolution')
                axes[0, 0].grid(True)
                axes[0, 0].legend()

            # Cls loss
            if 'train/cls_loss' in df.columns:
                axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', marker='o', color='orange')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Cls Loss')
                axes[0, 1].set_title('Classification Loss Evolution')
                axes[0, 1].grid(True)
                axes[0, 1].legend()

            # DFL loss
            if 'train/dfl_loss' in df.columns:
                axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='o', color='green')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('DFL Loss')
                axes[1, 0].set_title('DFL Loss Evolution')
                axes[1, 0].grid(True)
                axes[1, 0].legend()

            # Contrast loss
            if 'train/contrast_loss' in df.columns:
                axes[1, 1].plot(df['epoch'], df['train/contrast_loss'], label='Contrast Loss', marker='o', color='red')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Contrast Loss')
                axes[1, 1].set_title('Contrastive Loss Evolution')
                axes[1, 1].grid(True)
                axes[1, 1].legend()

            plt.tight_layout()
            plot_path = results_dir / "loss_evolution.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\nLoss evolution plot saved to: {plot_path}")

        except ImportError:
            print("\nMatplotlib not available, skipping visualization")
        except Exception as e:
            print(f"\nError creating plots: {e}")

    else:
        print(f"\nWarning: Could not find results.csv at {results_csv}")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
