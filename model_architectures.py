import tensorflow as tf
from tensorflow.keras import layers, Model

##########################
# DCGAN-STYLE GENERATOR
##########################
def build_dcgan_generator(latent_dim=100):
    """
    A DCGAN-like generator that upsamples from a 1D latent vector
    into a 256x256 RGB image.
    """
    model = tf.keras.Sequential(name="DCGAN_Generator")
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 512)))  # 8x8x512 feature map

    model.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

##########################
# DCGAN-STYLE DISCRIMINATOR
##########################
def build_dcgan_discriminator(input_shape=(256, 256, 3)):
    """
    A DCGAN-like discriminator that downsamples a 256x256 image
    to a single scalar real/fake prediction.
    """
    model = tf.keras.Sequential(name="DCGAN_Discriminator")
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(512, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

##########################
# U-NET GENERATOR (CycleGAN-Style)
##########################
def build_unet_generator(input_shape=(256, 256, 3)):
    """
    A simplified U-Net style generator for image-to-image translation
    (e.g., Monet style).
    """
    def conv_block(x, filters, kernel_size=4, strides=2, activation=True, norm=True, name='conv'):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name=f"{name}_conv")(x)
        if norm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        if activation:
            x = layers.LeakyReLU(0.2, name=f"{name}_lrelu")(x)
        return x

    def deconv_block(x, skip_input, filters, kernel_size=4, strides=2, name='deconv'):
        x = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same', name=f"{name}_deconv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dropout(0.5, name=f"{name}_drop")(x)
        x = layers.ReLU(name=f"{name}_relu")(x)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip_input])
        return x

    inputs = layers.Input(shape=input_shape, name='UNET_Input')

    # Downsampling (Encoder)
    c1 = conv_block(inputs, 64, name='enc1')
    c2 = conv_block(c1, 128, name='enc2')
    c3 = conv_block(c2, 256, name='enc3')
    c4 = conv_block(c3, 512, name='enc4')
    c5 = conv_block(c4, 512, name='enc5')
    c6 = conv_block(c5, 512, name='enc6')
    c7 = conv_block(c6, 512, name='enc7')
    # NOTE: remove strides=1 to ensure final shape is (1,1) if you want 8 levels
    c8 = conv_block(c7, 512, name='enc8')

    # Upsampling (Decoder)
    d1 = deconv_block(c8, c7, 512, name='dec1')
    d2 = deconv_block(d1, c6, 512, name='dec2')
    d3 = deconv_block(d2, c5, 512, name='dec3')
    d4 = deconv_block(d3, c4, 512, name='dec4')
    d5 = deconv_block(d4, c3, 256, name='dec5')
    d6 = deconv_block(d5, c2, 128, name='dec6')
    d7 = deconv_block(d6, c1, 64, name='dec7')

    outputs = layers.Conv2DTranspose(3, (4,4), strides=2, padding='same', activation='tanh', name='dec8_out')(d7)

    return Model(inputs=inputs, outputs=outputs, name="UNet_Generator")

##########################
# PATCHGAN DISCRIMINATOR
##########################
def build_patchgan_discriminator(input_shape=(256, 256, 3)):
    """
    A PatchGAN discriminator for image-to-image translation
    that outputs a grid of real/fake predictions.
    """
    def conv_block(x, filters, kernel_size=4, strides=2, activation=True, norm=True, name='conv'):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name=f"{name}_conv")(x)
        if norm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        if activation:
            x = layers.LeakyReLU(0.2, name=f"{name}_lrelu")(x)
        return x

    inp = layers.Input(shape=input_shape, name='disc_input')
    x = conv_block(inp, 64, name='disc_conv1')     
    x = conv_block(x, 128, name='disc_conv2')
    x = conv_block(x, 256, name='disc_conv3')
    x = conv_block(x, 512, strides=1, name='disc_conv4')

    out = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', name='disc_out')(x)
    return Model(inputs=inp, outputs=out, name="PatchGAN_Discriminator")

# Example usage (comment out if not needed):
if __name__ == "__main__":
    g_dcgan = build_dcgan_generator()
    d_dcgan = build_dcgan_discriminator()
    g_unet = build_unet_generator()
    d_patch = build_patchgan_discriminator()

    print("\n=== DCGAN Generator ===")
    g_dcgan.summary()

    print("\n=== DCGAN Discriminator ===")
    d_dcgan.summary()

    print("\n=== U-Net Generator ===")
    g_unet.summary()

    print("\n=== PatchGAN Discriminator ===")
    d_patch.summary()
