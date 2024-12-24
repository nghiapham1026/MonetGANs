import matplotlib.pyplot as plt
import tensorflow as tf

# Import or define your model builders
from model_architectures import (
     build_dcgan_generator, build_dcgan_discriminator,
     build_unet_generator, build_patchgan_discriminator
)

def compare_models():
    """
    Creates and plots a comparison of different model architectures
    based on layer count and total trainable parameters.
    """

    # Instantiate each model
    dcgan_gen = build_dcgan_generator(latent_dim=100)
    dcgan_disc = build_dcgan_discriminator(input_shape=(256, 256, 3))
    unet_gen = build_unet_generator(input_shape=(256, 256, 3))
    patch_disc = build_patchgan_discriminator(input_shape=(256, 256, 3))

    model_names = [
        "DCGAN Generator",
        "DCGAN Discriminator",
        "U-Net Generator",
        "PatchGAN Discriminator"
    ]
    models = [dcgan_gen, dcgan_disc, unet_gen, patch_disc]

    # Count layers & params
    num_layers = [len(m.layers) for m in models]
    num_params = [m.count_params() for m in models]

    # Print results
    for name, layers, params in zip(model_names, num_layers, num_params):
        print(f"Model: {name}")
        print(f"  Layers: {layers}")
        print(f"  Trainable Params: {params}\n")

    # Bar chart: number of layers
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, num_layers, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title("Comparison of Number of Layers")
    plt.xlabel("Models")
    plt.ylabel("Layer Count")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    # Bar chart: trainable params
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, num_params, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title("Comparison of Trainable Parameters")
    plt.xlabel("Models")
    plt.ylabel("Trainable Parameters")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    from model_architectures import (
        build_dcgan_generator, build_dcgan_discriminator,
        build_unet_generator, build_patchgan_discriminator
    )

    compare_models()
