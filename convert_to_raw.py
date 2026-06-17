#!/usr/bin/env python3
"""
convert_to_raw.py — Convert any image to raw grayscale (.raw) format.

Usage:
    python3 convert_to_raw.py input.jpg output.raw
    python3 convert_to_raw.py input.png output.raw --width 256 --height 256
    python3 convert_to_raw.py  (uses default: lena-like test pattern)

Output format:
    Exactly width*height bytes. Each byte is one pixel, 0=black, 255=white.
    No headers, no compression. Pass --width and --height to main.cpp.

Install dependencies:
    pip3 install Pillow numpy --break-system-packages
"""

import sys
import os
import argparse
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run:")
    print("  pip3 install Pillow numpy --break-system-packages")
    sys.exit(1)


def convert_image(input_path: str, output_path: str, width: int = None, height: int = None) -> tuple:
    """Convert image file to raw grayscale. Returns (width, height)."""
    img = Image.open(input_path).convert("L")  # grayscale
    if width and height:
        img = img.resize((width, height), Image.LANCZOS)
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)
    arr.tofile(output_path)
    print(f"Saved: {output_path}  ({w}x{h} = {w*h} bytes)")
    print(f"Run pipeline with:  W={w} H={h} IMG={output_path}")
    return w, h


def generate_test_pattern(output_path: str, width: int = 256, height: int = 256):
    """Generate a synthetic test image with visible edges."""
    arr = np.zeros((height, width), dtype=np.uint8)

    # White rectangle
    arr[40:height-40, 40:width-40] = 200

    # Inner black circle
    cy, cx = height // 2, width // 2
    r = min(height, width) // 4
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    arr[mask] = 30

    # Diagonal stripe
    for i in range(min(height, width)):
        if 0 <= i < height and 0 <= i < width:
            arr[i, i] = 255
        if 0 <= i < height and 0 <= i+1 < width:
            arr[i, i+1] = 255

    arr.tofile(output_path)
    print(f"Saved: {output_path}  ({width}x{height} = {width*height} bytes)")
    print(f"Run pipeline with:  W={width} H={height} IMG={output_path}")


def view_raw(raw_path: str, width: int, height: int):
    """View a .raw file (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        arr = np.fromfile(raw_path, dtype=np.uint8).reshape(height, width)
        plt.imshow(arr, cmap='gray')
        plt.title(os.path.basename(raw_path))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed. To view images, run:")
        print("  pip3 install matplotlib --break-system-packages")
        print("Or view with any raw image viewer (width x height, 8-bit grayscale).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to/from raw grayscale format for the RISC-V Canny pipeline."
    )
    parser.add_argument("input", nargs="?", help="Input image file (jpg/png/bmp/etc)")
    parser.add_argument("output", nargs="?", default="input.raw", help="Output .raw file (default: input.raw)")
    parser.add_argument("--width",  "-W", type=int, help="Resize to width")
    parser.add_argument("--height", "-H", type=int, help="Resize to height")
    parser.add_argument("--view",         action="store_true", help="View the raw file after creating it")
    parser.add_argument("--generate",     action="store_true", help="Generate a synthetic test pattern")
    args = parser.parse_args()

    if args.generate or args.input is None:
        w = args.width  or 256
        h = args.height or 256
        out = args.output or "input.raw"
        generate_test_pattern(out, w, h)
        if args.view:
            view_raw(out, w, h)
        return

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    w, h = convert_image(args.input, args.output, args.width, args.height)

    if args.view:
        view_raw(args.output, w, h)


if __name__ == "__main__":
    main()
