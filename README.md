# Monet-Style GAN Project

This repository showcases a mini-project to generate or translate images into **Claude Monet's** painterly style using **Generative Adversarial Networks (GANs)**. It explores data, defines multiple GAN architectures (DCGAN and U-Net + PatchGAN), compares them, and demonstrates hyperparameter tuning.

## Overview

1. **Data Exploration** (`data_exploration.py`):
   - Loads Monet and Photo datasets.
   - Checks validity and dimensions of images (256×256).
   - Provides a quick histogram visualization function for color channels (RGB).

2. **Model Architectures** (`model_architectures.py`):
   - Implements both **DCGAN** (generator and discriminator) and **U-Net + PatchGAN** for image-to-image translation.
   - DCGAN is useful for unconditional image generation (from random noise).
   - U-Net + PatchGAN is designed for translating real photos into Monet-style images (image-to-image translation).

3. **Architecture Comparison** (`compare_architectures.py`):
   - Instantiates each model and compares:
     - Number of layers
     - Trainable parameters
   - Visualizes these comparisons in bar charts.

4. **Hyperparameter Experiments** (`hyperparameter_experiments.py`):
   - Demonstrates a simplified hyperparameter-tuning approach.
   - Varies learning rates, batch sizes, and architectures.
   - Returns dummy losses and plots them to illustrate how you might track and compare results in real training.

---

## Repository Structure

```
monet-gan-project/
├── data_exploration.py
├── model_architectures.py
├── compare_architectures.py
├── hyperparameter_experiments.py
├── README.md
└── (Other files like .gitignore, etc.)
```

### data_exploration.py
- **Key Functions**:
  - `explore_data(monet_dir, photo_dir)`: Prints image counts, valid/invalid checks, and sample metadata.
  - `plot_color_histogram(image_path)`: Plots a color histogram (R/G/B) for a single image.
- **Usage**: 
  ```bash
  python data_exploration.py
  ```
  Adjust the directory paths in the `__main__` section as needed.

### model_architectures.py
- Contains **DCGAN** and **U-Net + PatchGAN** definitions.
- **Usage**:  
  You can import these functions into your training scripts:
  ```python
  from model_architectures import (
      build_dcgan_generator, 
      build_dcgan_discriminator,
      build_unet_generator, 
      build_patchgan_discriminator
  )
  ```
- You can also run the file directly to print model summaries:
  ```bash
  python model_architectures.py
  ```

### compare_architectures.py
- Builds each model and compares:
  - Layer count
  - Number of parameters
- Plots bar charts for easy visualization of complexity.

### hyperparameter_experiments.py
- Provides a **dummy** example of how you might perform hyperparameter searches:
  - Varies generator/discriminator learning rates, batch sizes, etc.
  - Demonstrates tracking final losses and plotting them.
- **Usage**:
  ```bash
  python hyperparameter_experiments.py
  ```

---

## Getting Started

1. **Install Dependencies**  
   - [Python 3.x](https://www.python.org/downloads/)  
   - [TensorFlow 2.x](https://www.tensorflow.org/install)  
   - `pip install tensorflow matplotlib pillow`

2. **Project Setup**  
   - Clone or download this repository.
   - Place your Monet (`monet_jpg`) and photo (`photo_jpg`) directories in the root (or change the paths in the scripts).

3. **Run Data Exploration**  
   ```bash
   python data_exploration.py
   ```
   - Adjust `monet_dir` and `photo_dir` in the script if needed.
   - Observe image counts and sample histograms.

4. **Check Model Architectures**  
   ```bash
   python model_architectures.py
   ```
   - View the DCGAN and U-Net + PatchGAN summaries.

5. **Compare Architectures**  
   ```bash
   python compare_architectures.py
   ```
   - See a bar chart comparing layers and parameter sizes.

6. **Hyperparameter Experiments**  
   ```bash
   python hyperparameter_experiments.py
   ```
   - Explore how different learning rates or batch sizes might affect training (dummy metrics for demonstration).

---

## Notes and Future Work

- **Real Dataset**: For a complete project, integrate your training loops with real data, possibly using libraries like [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data).
- **Metrics**: Implement advanced GAN evaluation metrics (FID, Inception Score, etc.) to measure image quality more objectively.
- **Longer Training**: Adjust epochs, use GPU/TPU to handle computational loads.