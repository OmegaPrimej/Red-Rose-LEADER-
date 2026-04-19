#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def generate_vessel(size_kb, output):
    """Generate random binary data for the neural trainer."""
    data = os.urandom(size_kb * 1024)
    Path(output).write_bytes(data)
    print(f"Generated {size_kb} KB vessel: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test vessel for Queen trainer")
    parser.add_argument("--size", type=int, default=64, help="Size in KB")
    parser.add_argument("--output", default="queentarget.bin", help="Output file")
    args = parser.parse_args()
    
    generate_vessel(args.size, args.output)
