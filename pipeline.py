#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Central orchestrator for the SSEC Semiconductor modeling pipeline.

Usage:
  python pipeline.py --mode data
  python pipeline.py --mode piae
  python pipeline.py --mode cvae
  python pipeline.py --mode interp --model-type piae --model-path outputs/checkpoints/piae_v3_s21_final.pt
  python pipeline.py --mode all

Modes:
  data   : Run PCA pipeline and split generations
  piae   : Train Physics-Informed Autoencoder
  cvae   : Train Physics-Supervised Conditional VAE
  interp : Run interpretability methods on a trained model
  all    : Run data, then piae, then cvae
"""

import os
import sys
import argparse

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import data_processing
import piae_model
import cvae_model
import interpretability

def main():
    parser = argparse.ArgumentParser(description="SSEC Semiconductor Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['data', 'piae', 'cvae', 'interp', 'all'],
                        help="Pipeline stage to execute.")
    parser.add_argument('--model-type', type=str, choices=['piae', 'cvae'], default='piae',
                        help="Model type for interpretability mode.")
    parser.add_argument('--model-path', type=str, default='',
                        help="Path to the checkpoint for interpretability mode.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for training.")
    
    args = parser.parse_args()

    # Paths (relative to project root)
    paths = {
        'data_dir':   "data/raw",
        'splits_dir': "data/splits",
        'pca_dir':    "data/pca_artifacts",
        'proc_dir':   "data/processed",
    }

    if args.mode in ['data', 'all']:
        print("\n" + "="*50)
        print("🚀 STAGE 1: DATA PROCESSING")
        print("="*50)
        data_processing.run(
            data_dir=paths['data_dir'],
            splits_dir=paths['splits_dir'],
            pca_dir=paths['pca_dir'],
            proc_dir=paths['proc_dir'],
            plot_dir="outputs/plots/data"
        )

    if args.mode in ['piae', 'all']:
        print("\n" + "="*50)
        print("🚀 STAGE 2: PIAE MODEL TRAINING")
        print("="*50)
        piae_model.run(
            pca_dir=paths['pca_dir'],
            proc_dir=paths['proc_dir'],
            splits_dir=paths['splits_dir'],
            plot_dir="outputs/plots/piae",
            ckpt_dir="outputs/checkpoints",
            epochs=args.epochs,
            lr=args.lr
        )

    if args.mode in ['cvae', 'all']:
        print("\n" + "="*50)
        print("🚀 STAGE 3: CVAE MODEL TRAINING")
        print("="*50)
        cvae_model.run(
            pca_dir=paths['pca_dir'],
            proc_dir=paths['proc_dir'],
            splits_dir=paths['splits_dir'],
            plot_dir="outputs/plots/cvae",
            ckpt_dir="outputs/checkpoints",
            epochs=args.epochs,
            lr=args.lr
        )

    if args.mode == 'interp':
        print("\n" + "="*50)
        print(f"🚀 STAGE 4: INTERPRETABILITY ({args.model_type.upper()})")
        print("="*50)
        if not args.model_path:
            # Default fallback paths
            if args.model_type == 'piae':
                args.model_path = "outputs/checkpoints/piae_v3_s21_final.pt"
            else:
                args.model_path = "outputs/checkpoints/cvae_v5_physics_supervised_final.pt"
            print(f"No --model-path provided, defaulting to: {args.model_path}")

        interpretability.run(
            model_path=args.model_path,
            model_type=args.model_type,
            data_dir=paths['data_dir'],
            pca_dir=paths['pca_dir'],
            proc_dir=paths['proc_dir'],
            splits_dir=paths['splits_dir'],
            plot_dir="outputs/plots/interp"
        )

if __name__ == "__main__":
    main()
