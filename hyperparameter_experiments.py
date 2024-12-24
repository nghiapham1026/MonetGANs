import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from model_architectures import (
    build_dcgan_generator, build_dcgan_discriminator,
    build_unet_generator, build_patchgan_discriminator
)

def train_gan(config, epochs=2):
    """
    Trains a GAN with dummy logic for demonstration.
    You can replace with actual data + training loops.
    """
    arch = config["architecture"]
    gen_lr = config["gen_lr"]
    disc_lr = config["disc_lr"]
    batch_size = config["batch_size"]

    # Build models
    if arch == "dcgan":
        generator = build_dcgan_generator(latent_dim=100)
        discriminator = build_dcgan_discriminator()
    elif arch == "unet_patchgan":
        generator = build_unet_generator()
        discriminator = build_patchgan_discriminator()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5)

    print(f"\nTraining config: {config}")
    print(f"Generator: {generator.name}, Discriminator: {discriminator.name}")
    print(f"gen_lr={gen_lr}, disc_lr={disc_lr}, batch_size={batch_size}, epochs={epochs}")

    # Simulate training time
    time.sleep(1.0)

    # Return dummy metrics
    np.random.seed()
    final_gen_loss = float(np.random.uniform(0.2, 1.5))
    final_disc_loss = float(np.random.uniform(0.2, 1.5))
    return {
        "final_gen_loss": final_gen_loss,
        "final_disc_loss": final_disc_loss
    }

def run_experiments():
    """
    Runs several hyperparameter configs and plots the final gen/disc loss.
    """

    hyperparameter_configs = [
        {"architecture": "dcgan", "gen_lr": 2e-4, "disc_lr": 2e-4, "batch_size": 16},
        {"architecture": "dcgan", "gen_lr": 1e-4, "disc_lr": 4e-4, "batch_size": 32},
        {"architecture": "unet_patchgan", "gen_lr": 2e-4, "disc_lr": 2e-4, "batch_size": 1},
        {"architecture": "unet_patchgan", "gen_lr": 1e-4, "disc_lr": 1e-4, "batch_size": 2},
    ]

    all_results = []
    for cfg in hyperparameter_configs:
        metrics = train_gan(cfg, epochs=2)
        combined = {**cfg, **metrics}
        all_results.append(combined)

    print("\n=== ALL EXPERIMENT RESULTS ===")
    for r in all_results:
        print(r)

    # Visualization
    labels = []
    gen_losses = []
    disc_losses = []
    for r in all_results:
        label = (f"{r['architecture']} | gen_lr={r['gen_lr']}, "
                 f"disc_lr={r['disc_lr']}, bs={r['batch_size']}")
        labels.append(label)
        gen_losses.append(r["final_gen_loss"])
        disc_losses.append(r["final_disc_loss"])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, gen_losses, width, label='Gen Loss', color='skyblue')
    plt.bar(x + width/2, disc_losses, width, label='Disc Loss', color='salmon')
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.xlabel("Experiment Config")
    plt.ylabel("Dummy Final Loss")
    plt.title("Comparison of Hyperparameter Experiments")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    from model_architectures import (
        build_dcgan_generator, build_dcgan_discriminator,
        build_unet_generator, build_patchgan_discriminator
    )

    run_experiments()
