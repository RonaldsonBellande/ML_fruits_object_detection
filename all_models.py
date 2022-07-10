from header_imports import *

class models(object):
    def create_models_1(self):

        model = Sequential()
        model.add(Conv2D(filters=64,kernel_size=(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=16,kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        model.add(MaxPooling2D(pool_size = (1,1)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    # UNET model
    def unet_model(self):
        inputs = keras.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        ### [First half of the network: downsampling inputs] ###
        x = layers.Conv2D(32, 3, strides=2, padding="same")(augmented)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        ### [Second half of the network: upsampling inputs] ###
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        outputs = layers.Conv2D(self.number_classes, 3, activation="softmax", padding="same")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)
        
        return model



    # VIT Transformer model 
    def vit_transformer_shift_model(self):

        inputs = layers.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        patches = Patches()(augmented)
        encoded_patches = PatchEncoder()(patches)
        (shift_patches, _) = ShiftedPatchTokenization()(encoded_patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=self.epsilon)(shift_patches)
            attention_output = MultiHeadAttentionLSA(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1, attention_mask=self.diag_attn_mask)
            x2 = layers.Add()([attention_output, shift_patches])
            x3 = layers.LayerNormalization(epsilon=self.epsilon)(x2)
            x3 = self.multilayer_perceptron(x3, self.transformer_units, 0.1)
            shift_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=self.epsilon)(shift_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.multilayer_perceptron(representation, self.mlp_head_units, 0.5)
        outputs = layers.Dense(self.number_classes)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)

        return model


    # VIT Transformer model 
    def vit_transformer_shift_noise_model(self):

        inputs = layers.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        patches = Patches()(augmented)
        encoded_patches = PatchEncoder()(patches)
        (shift_patches, _) = ShiftedPatchTokenization()(encoded_patches)
        (noise_patches, _) = RandomPatchNoise()(shift_patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=self.epsilon)(noise_patches)
            attention_output = MultiHeadAttentionLSA(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1, attention_mask=self.diag_attn_mask)
            x2 = layers.Add()([attention_output, noise_patches])
            x3 = layers.LayerNormalization(epsilon=self.epsilon)(x2)
            x3 = self.multilayer_perceptron(x3, self.transformer_units, 0.1)
            noise_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=self.epsilon)(noise_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.multilayer_perceptron(representation, self.mlp_head_units, 0.5)
        outputs = layers.Dense(self.number_classes)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)
        
        return model


    def multilayer_perceptron(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


    # CNN with LSTM models
    def cnn_lstm_model(self):

        input = layers.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu",)(augmented)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu",)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding="same", return_sequences=True, activation="relu",)(x)
        x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)

        model = keras.models.Model(input, x)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)

        return model



    # Personal model
    def personal_model(self):

        inputs = layers.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        # Transformer 
        patches = Patches()(inputs)
        encoded_patches = PatchEncoder()(patches)
        (shift_patches, _) = ShiftedPatchTokenization()(encoded_patches)
        (noise_patches, _) = RandomPatchNoise()(shift_patches)

        # None-Transformer
        shift = ShiftedTokenization()(inputs)
        noise = RandomNoise()(shift)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=self.epsilon)(encoded_patches)
            attention_output = MultiHeadAttentionLSA(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1, attention_mask=self.diag_attn_mask)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=self.epsilon)(x2)
            x3 = self.multilayer_perceptron(x3, self.transformer_units, 0.1)
            encoded_patches = layers.Add()([x3, x2])

        ### [First half of the network: downsampling inputs] ###
        x = layers.Conv2D(32, 3, strides=2, padding="same")(augmented)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        ### [Second half of the network: upsampling inputs] ###
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=self.epsilon)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.multilayer_perceptron(representation, self.mlp_head_units, 0.5)


        for filters in [int(self.number_classes * 2)]:
            x = layers.Conv2D(filters, 3, activation="softmax", padding="same")(x)
            x2 = layers.Dense(filters)(features)
            x = layers.add([x, x2])  # Add back residual

            x = layers.Activation("relu")(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", activation="relu")(residual)
            x = layers.add([x, residual])  # Add back residual


        x = layers.Conv2D(self.number_classes, 3, activation="softmax", padding="same")(x)
        outputs = layers.Dense(self.number_classes)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)

        return model




    # Personal model for video
    def personal_model_2(self):

        inputs = layers.Input(shape=self.input_shape)
        augmented = self.augmentation(inputs)

        # Transformer 
        patches = Patches()(inputs)
        encoded_patches = PatchEncoder()(patches)
        (shift_patches, _) = ShiftedPatchTokenization()(encoded_patches)
        (noise_patches, _) = RandomPatchNoise()(shift_patches)

        # None-Transformer
        shift = ShiftedTokenization()(inputs)
        # noise = RandomNoise()(shift)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=self.epsilon)(encoded_patches)
            attention_output = MultiHeadAttentionLSA(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1, x1, attention_mask=self.diag_attn_mask)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=self.epsilon)(x2)
            x3 = self.multilayer_perceptron(x3, self.transformer_units, 0.1)
            encoded_patches = layers.Add()([x3, x2])

        ### [First half of the network: downsampling inputs]
        x = layers.ConvLSTM2D(32, 3, strides=2, padding="same", activation="relu", return_sequences=True)(shift)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.separableconv2d(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.ConvLSTM2D(filters, 1, strides=2, padding="same", activation="relu", return_sequences=True)(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        ### [Second half of the network: upsampling inputs] ###
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.ConvLSTM2D(filters, 1, padding="same", activation="relu", return_sequences=True)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=self.epsilon)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.multilayer_perceptron(representation, self.mlp_head_units, 0.5)


        for filters in [int(self.number_classes * 2)]:
            x = layers.ConvLSTM2D(filters, 3, activation="softmax", padding="same", return_sequences=True)(x)
            x2 = layers.Dense(filters)(features)
            x = layers.add([x, x2])  # Add back residual

            x = layers.Activation("relu")(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.ConvLSTM2D(filters, 1, padding="same", activation="relu", return_sequences=True)(residual)
            x = layers.add([x, residual])  # Add back residual


        x = layers.ConvLSTM2D(self.number_classes, 3, activation="softmax", padding="same", return_sequences=True)(x)
        outputs = layers.Dense(self.number_classes)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),)

        return model

