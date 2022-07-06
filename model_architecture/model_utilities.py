from header_imports import *


class model_utilities(object):
    def __init__(self):    
        
        self.image_size = 240
        self.number_of_nodes = 16
        
        # Transformer
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [self.projection_dim * 2, self.projection_dim]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]
        self.epsilon = 1e-6
        
        self.augmentation = keras.Sequential([
            layers.Normalization(),
            layers.Resizing(self.image_size, self.image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ])

        self.diag_attn_mask = tf.cast([(1-tf.eye(self.num_patches))], dtype=tf.int8)
        self.random_noise_count = 1


class Patches(layers.Layer, model_utilities):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_utilities.__init__(self)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer, model_utilities):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_utilities.__init__(self)
        
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class MultiHeadAttentionLSA(layers.MultiHeadAttention, model_utilities):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_utilities.__init__(self)

        self.tau = tf.cast(tf.Variable(math.sqrt(float(self._key_dim)), trainable=True), tf.float16)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)

        return attention_output, attention_scores



class ShiftedPatchTokenization(layers.Layer, model_utilities):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_utilities.__init__(self)
        
        self.half_patch = self.patch_size // 2
        self.flatten_patches = layers.Reshape((self.num_patches, -1))
        self.projection = layers.Dense(units=self.projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)

    def crop_shift_pad(self, images, shift):
        # Build the diagonally shifted images
        if shift == "normal":
            crop_height = 0
            crop_width = 0
            shift_height = 0
            shift_width = 0
        elif shift == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif shift == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif shift == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        elif shift == "right-down":
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )

        shift_pad = tf.expand_dims(shift_pad, 0)
        return shift_pad

    def call(self, images):
        images = tf.concat(
            [
                self.crop_shift_pad(images, shift="normal"),
                self.crop_shift_pad(images, shift="left-up"),
                self.crop_shift_pad(images, shift="left-down"),
                self.crop_shift_pad(images, shift="right-up"),
                self.crop_shift_pad(images, shift="right-down"),
            ],
            axis=-1,
        )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        flat_patches = self.flatten_patches(patches)
        
        # Layer normalize the flat patches and linearly project it
        tokens = self.layer_norm(flat_patches)
        tokens = self.projection(tokens)

        return (tokens, patches)



class RandomPatchNoise(layers.Layer, model_utilities):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_utilities.__init__(self)

        self.half_patch = self.patch_size // 2
    
    def adding_random_noise(self, image, noise_type):
     
        images = image
        if noise_type == "Normal":
            images = image

        elif noise_type == "Gaussian":
            # Gaussian noise
            for i in range(self.random_noise_count):
                gaussian_noise = np.random.normal(0, (10 **0.5), image.shape)
                images = image + gaussian_noise

        elif noise_type == "SaltPepper":
            # Salt and pepper noise
            for i in range(self.random_noise_count):
                probability = 0.02
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        random_num = random.random()
                        if random_num < probability:
                            images[i][j] = 0
                        elif random_num > (1 - probability):
                            images[i][j] = 255

        elif noise_type == "Poisson":
            # Poisson noise
            for i in range(self.random_noise_count):
                poisson_noise = np.sqrt(image) * np.random.normal(0, 1, image.shape)
                images = image + poisson_noise

        elif noise_type == "Speckle":
            # Speckle noise
            for i in range(self.random_noise_count):
                speckle_noise = np.random.normal(0, (10 **0.5), image.shape)
                images = image + image * speckle_noise

        elif noise_type == "Uniform":
            # Uniform noise
            for i in range(self.random_noise_count):
                uniform_noise = np.random.uniform(0,(10 **0.5), image.shape)
                images = image + uniform_noise


        random_img = tf.expand_dims(images, 0)
        return random_img

    def call(self, images):
        images = tf.concat(
            [
                self.adding_random_noise(images, noise_type="normal"),
                self.adding_random_noise(images, noise_type="Gaussian"),
                # self.adding_random_noise(images, noise_type="SaltPepper"),
                # self.adding_random_noise(images, noise_type="Poisson"),
                self.adding_random_noise(images, noise_type="Speckle"),
                self.adding_random_noise(images, noise_type="Uniform"),
            ],
            axis=-1,
        )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = keras.layers.Flatten(data_format=patches)
       
        # Layer normalize the flat patches and linearly project it
        tokens = self.layer_norm(flat_patches)
        tokens = self.projection(tokens)

        return (tokens, patches)

