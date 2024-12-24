import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def explore_data(monet_dir, photo_dir):
    """
    Basic data exploration for Monet and Photo directories.
    Prints counts, checks image validity/size, shows sample metadata.
    """

    # Gather image paths
    monet_paths = sorted(glob.glob(os.path.join(monet_dir, '*.jpg')))
    photo_paths = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')))

    print("Number of Monet images:", len(monet_paths))
    print("Number of Photo images:", len(photo_paths))

    def get_image_metadata(image_path):
        """
        Returns basic metadata for an image: width, height, mode, valid.
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'path': image_path,
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'valid': True
                }
        except Exception:
            # If there's an error reading the image, mark as invalid
            return {
                'path': image_path,
                'width': None,
                'height': None,
                'mode': None,
                'valid': False
            }

    # Show a small sample of Monet images
    print("\n=== Sample Monet Images ===")
    sample_monet = random.sample(monet_paths, min(5, len(monet_paths)))
    for path in sample_monet:
        meta = get_image_metadata(path)
        print(meta)

    # Show a small sample of Photo images
    print("\n=== Sample Photo Images ===")
    sample_photos = random.sample(photo_paths, min(5, len(photo_paths)))
    for path in sample_photos:
        meta = get_image_metadata(path)
        print(meta)

    # Example of removing or listing images that are invalid or not 256x256
    def clean_image_paths(image_paths, required_w=256, required_h=256):
        valid_paths = []
        for img_path in image_paths:
            meta = get_image_metadata(img_path)
            if meta['valid'] and meta['width'] == required_w and meta['height'] == required_h:
                valid_paths.append(img_path)
        return valid_paths

    monet_clean = clean_image_paths(monet_paths)
    photo_clean = clean_image_paths(photo_paths)

    print(f"\nValid Monet images (256x256): {len(monet_clean)}")
    print(f"Valid Photo images (256x256): {len(photo_clean)}")

def plot_color_histogram(image_path):
    """
    Plots the R/G/B histogram for a single image.
    """
    with Image.open(image_path) as img:
        # Convert to RGB if not already
        img = img.convert('RGB')
        pixels = np.array(img).reshape(-1, 3)

        plt.figure(figsize=(8, 4))
        plt.suptitle(f"Color Histogram\n{os.path.basename(image_path)}", fontsize=14)

        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.hist(pixels[:, i], bins=50, color=color, alpha=0.5, label=color)

        plt.xlabel("Pixel Intensity")
        plt.ylabel("Count")
        plt.legend()
        plt.show()


# =====================
# Expanded Example Usage
# =====================
if __name__ == "__main__":
    # Provide the correct local paths to your Monet and Photo directories
    monet_dir = "monet_jpg"  # Update if needed
    photo_dir = "photo_jpg"  # Update if needed

    # 1. Run a basic data exploration
    explore_data(monet_dir, photo_dir)

    # 2. Demonstrate how to plot color histograms for random images
    monet_paths = sorted(glob.glob(os.path.join(monet_dir, '*.jpg')))
    photo_paths = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')))

    if monet_paths:
        random_monet = random.choice(monet_paths)
        print(f"\nPlotting color histogram for a random Monet painting: {random_monet}")
        plot_color_histogram(random_monet)

    if photo_paths:
        random_photo = random.choice(photo_paths)
        print(f"\nPlotting color histogram for a random Photo: {random_photo}")
        plot_color_histogram(random_photo)
