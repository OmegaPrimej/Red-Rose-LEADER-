#!/usr/bin/env python3
"""
Apply post-hoc fixes to saved model checkpoints.
"""
import argparse
import numpy as np
from pathlib import Path
from queencore import QueenAuoraBigram


def heal_checkpoint(input_path, output_path):
    """Load, validate, and heal a model checkpoint."""
    data = np.load(input_path)
    
    # Create model and restore weights
    model = QueenAuoraBigram()
    model.emb = data["emb"]
    model.Wh = data["Wh"] 
    model.Wout = data["Wout"]
    
    # Basic sanity checks and repairs
    if np.any(np.isnan(model.Wout)):
        print("NaN weights detected, zeroing...")
        model.Wout = np.nan_to_num(model.Wout)
    
    if np.linalg.norm(model.Wout) > 100:
        print("Exploding gradients detected, clipping...")
        model.Wout = np.clip(model.Wout, -5, 5)
    
    # Save healed model
    np.savez_compressed(output_path, emb=model.emb, Wh=model.Wh, Wout=model.Wout)
    print(f"Healed checkpoint saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heal corrupted model checkpoints")
    parser.add_argument("input", help="Input .npz checkpoint")
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args()
    
    output = args.output or f"healed_{Path(args.input).name}"
    heal_checkpoint(args.input, output)
