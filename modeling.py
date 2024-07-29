import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Dense, BatchNormalization, Dropout, Flatten, AveragePooling2D, \
    Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class JerseyNumberRecognition:
    def __init__(self):
        self.model = self.create_model(nb_classes_0=11, nb_classes_1=11, img_rows=64, img_cols=64, img_channels=3,
                                  nb_filters=32,
                                  pool_size=(2, 2), kernel_size=(3, 3))

    @staticmethod
    def create_model(nb_classes_0=11, nb_classes_1=11, img_rows=64, img_cols=64, img_channels=3, nb_filters=32,
                     pool_size=(2, 2), kernel_size=(3, 3)):
        inputs = Input(shape=(img_rows, img_cols, img_channels))

        # Convolutional layers
        cov = Conv2D(nb_filters, kernel_size=kernel_size, padding="same")(inputs)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = Conv2D(nb_filters, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = AveragePooling2D(pool_size=pool_size)(cov)
        cov = Dropout(0.3)(cov)

        cov = Conv2D(nb_filters * 2, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = Conv2D(nb_filters * 2, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = AveragePooling2D(pool_size=pool_size)(cov)
        cov = Dropout(0.3)(cov)

        cov = Conv2D(nb_filters * 4, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = Conv2D(nb_filters * 4, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = AveragePooling2D(pool_size=pool_size)(cov)
        cov = Dropout(0.3)(cov)

        cov = Conv2D(nb_filters * 8, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = Conv2D(nb_filters * 8, kernel_size=kernel_size, padding="same")(cov)
        cov = BatchNormalization()(cov)
        cov = Activation('relu')(cov)
        cov = AveragePooling2D(pool_size=pool_size)(cov)
        cov = Dropout(0.3)(cov)

        cov_out = Flatten()(cov)

        cov_bin = Dense(128, activation='relu')(cov_out)
        cov_bin = BatchNormalization()(cov_bin)
        cov_bin = Dropout(0.3)(cov_bin)
        is_two_digits = Dense(3, activation='softmax', name="is_two_digits")(cov_bin)

        # Fully connected layers for digit 0
        cov2 = Dense(128, activation='relu')(cov_out)
        cov2 = BatchNormalization()(cov2)
        cov2 = Dropout(0.3)(cov2)
        cov2 = Dense(64, activation='relu')(cov2)
        cov2 = BatchNormalization()(cov2)
        cov2 = Dropout(0.3)(cov2)
        cov2 = Dense(64, activation='relu')(cov2)
        cov2 = BatchNormalization()(cov2)
        cov2 = Dropout(0.3)(cov2)
        c0 = Dense(nb_classes_0, activation='softmax', name="digit_0")(cov2)

        # Fully connected layers for digit 1
        cov3 = Dense(128, activation='relu')(cov_out)
        cov3 = BatchNormalization()(cov3)
        cov3 = Dropout(0.3)(cov3)
        cov3 = Dense(64, activation='relu')(cov3)
        cov3 = BatchNormalization()(cov3)
        cov3 = Dropout(0.3)(cov3)
        cov3 = Dense(64, activation='relu')(cov3)
        cov3 = BatchNormalization()(cov3)
        cov3 = Dropout(0.3)(cov3)
        c1 = Dense(nb_classes_1, activation='softmax', name="digit_1")(cov3)

        # Defining the model
        model = Model(inputs=inputs, outputs={"is_two_digits": is_two_digits, "digit_0": c0, "digit_1": c1})
        # model = Model(inputs=inputs, outputs=[is_two_digits,c0,c1])
        return model

    def compile_model(self):
        self.model.compile(optimizer=Adam(),
                           loss={'is_two_digits': 'categorical_crossentropy', 'digit_0': 'categorical_crossentropy',
                                 'digit_1': 'categorical_crossentropy'},
                           metrics={'is_two_digits': 'accuracy', 'digit_0': 'accuracy', 'digit_1': 'accuracy'})

    def train_model_generator(self, train_gen, valid_gen, training_steps, checkpoint_filepath, validation_steps,
                              epochs=50, batch_size=32, verbose=1):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks = [model_checkpoint_callback]

        history = self.model.fit(
            train_gen,  # Training generator
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            steps_per_epoch=training_steps,
            validation_data=valid_gen,  # Validation generator
            validation_steps=validation_steps,  # Number of validation steps
            callbacks=callbacks
        )
        return history
