"""
AC-GAN (Auxiliary Classifier Generative Adversarial Network) Architecture.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class ChestXRayACGAN:
    """
    Conditional AC-GAN for multi-class medical image synthesis with auxiliary classification.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 2,
        image_shape: tuple = (64, 64, 3),
        kernel_size: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 6e-9,
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.generator = None
        self.discriminator = None
        self.gan = None

    def build_generator(self) -> models.Model:
        """
        Builds the Generator model that maps (noise, class_label) to a synthetic image.
        """
        noise_input = layers.Input(shape=(self.latent_dim,), name="noise_input")
        label_input = layers.Input(shape=(self.num_classes,), name="label_input")

        # Conditioning: Concatenate latent vector and one-hot class label
        x = layers.concatenate([noise_input, label_input])
        x = layers.Dense(1024, name="gen_dense_1")(x)
        x = layers.Dense(
            8 * 8 * 256,
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="gen_dense_2",
        )(x)
        x = layers.Reshape((8, 8, 256), name="gen_reshape")(x)

        filters = [256, 128, 64, 32]
        for idx, num_filters in enumerate(filters):
            strides = 2 if num_filters >= 64 else 1
            x = layers.LayerNormalization(name=f"gen_ln_{idx}")(x)
            x = layers.Activation("relu", name=f"gen_relu_{idx}")(x)
            x = layers.Conv2DTranspose(
                num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                strides=strides,
                name=f"gen_deconv_{idx}",
            )(x)

        # Output projection to RGB image with sigmoid activation [0, 1]
        x = layers.Conv2DTranspose(
            self.image_shape[-1],
            kernel_size=self.kernel_size,
            padding="same",
            name="gen_deconv_out",
        )(x)
        output_image = layers.Activation("sigmoid", name="gen_output")(x)

        model = models.Model(
            inputs=[noise_input, label_input],
            outputs=output_image,
            name="generator",
        )
        self.generator = model
        return model

    def build_discriminator(self) -> models.Model:
        """
        Builds the Discriminator model with dual output heads:
        1. Validity output (Real vs. Fake score)
        2. Classification output (Diagnostic label prediction)
        """
        image_input = layers.Input(shape=self.image_shape, name="disc_image_input")
        x = image_input

        filters = [32, 64, 128, 256]
        for idx, num_filters in enumerate(filters):
            strides = 2 if num_filters < 256 else 1
            x = layers.Conv2D(
                num_filters,
                kernel_size=self.kernel_size,
                padding="same",
                strides=strides,
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
                name=f"disc_conv_{idx}",
            )(x)
            x = layers.LeakyReLU(alpha=0.2, name=f"disc_leaky_{idx}")(x)

        x = layers.Flatten(name="disc_flatten")(x)

        # Head 1: Real / Fake validity score
        validity_out = layers.Dense(1, name="validity_output")(x)

        # Head 2: Auxiliary multi-class classification
        aux_dense = layers.Dense(
            256,
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="disc_aux_dense",
        )(x)
        aux_drop = layers.Dropout(0.3, name="disc_aux_dropout")(aux_dense)
        aux_logits = layers.Dense(self.num_classes, name="disc_aux_logits")(aux_drop)
        class_out = layers.Activation("softmax", name="class_output")(aux_logits)

        model = models.Model(
            inputs=image_input,
            outputs=[validity_out, class_out],
            name="discriminator",
        )
        self.discriminator = model
        return model

    def compile_models(self):
        """Compiles Discriminator and Combined AC-GAN models with RMSprop optimizers."""
        if self.generator is None:
            self.build_generator()
        if self.discriminator is None:
            self.build_discriminator()

        # Compile Discriminator independently
        disc_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        self.discriminator.compile(
            loss=["mse", "binary_crossentropy"],
            optimizer=disc_optimizer,
            metrics={"class_output": "accuracy"},
        )

        # Freeze Discriminator inside combined GAN graph
        self.discriminator.trainable = False

        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        generated_img = self.generator([noise_input, label_input])
        validity, classification = self.discriminator(generated_img)

        self.gan = models.Model(
            inputs=[noise_input, label_input],
            outputs=[validity, classification],
            name="acgan_combined",
        )

        gan_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate * 0.5,
            weight_decay=self.weight_decay * 0.5,
        )
        self.gan.compile(
            loss=["mse", "binary_crossentropy"],
            optimizer=gan_optimizer,
        )

        return self.generator, self.discriminator, self.gan
