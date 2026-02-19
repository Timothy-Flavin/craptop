import numpy as np
from PIL import Image

def convert_png_to_bin(image_path, output_path):
    # 1. Load image and convert to Grayscale ('L')
    img = Image.open(image_path).convert('L')
    
    # 2. Resize to 32x32 using Lanczos for high-quality downsampling
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    
    # 3. Convert to numpy array and normalize to [-1.0, 1.0]
    # PNG pixels are 0-255; dividing by 255.0 then scaling to [-1, 1]
    grid = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    
    # 4. Flatten to a 1D array
    grid_1d = grid.flatten()
    
    # 5. Save as raw binary file
    grid_1d.tofile(output_path)
    print(f"Success! Saved {len(grid_1d)} floats to {output_path}")

if __name__ == "__main__":
    fin = input("Enter path to input PNG image (e.g., 'map0.png'): ")
    fout = input("Enter path for output binary file (e.g., 'map.bin'): ")
    convert_png_to_bin(fin, fout)