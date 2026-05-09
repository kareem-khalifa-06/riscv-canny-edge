from PIL import Image
import sys

if len(sys.argv) != 4:
    print("Usage: python3 convert_to_raw.py <input.jpg/png> <width> <height>")
    sys.exit(1)

in_path = sys.argv[1]
w = int(sys.argv[2])
h = int(sys.argv[3])

img = Image.open(in_path).convert('L')  # Convert to grayscale
img = img.resize((w, h))                # Resize to target dimensions
img.tofile(in_path.rsplit('.', 1)[0] + '.raw')
print(f"Created {in_path.rsplit('.', 1)[0]}.raw ({w}x{h})")
