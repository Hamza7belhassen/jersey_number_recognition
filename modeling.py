import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Activation, Dense, BatchNormalization, Dropout, Flatten, AveragePooling2D, \
    Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_generation2 import load_dataset_soccernet_m , designate_batches , DataGenerator
from train import train_model_generator



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
    model = Model(inputs=inputs, outputs={"is_two_digits":is_two_digits,"digit_0": c0,"digit_1": c1})
    # model = Model(inputs=inputs, outputs=[is_two_digits,c0,c1])
    return model


#
# model = create_model(nb_classes_0=11, nb_classes_1=11, img_rows=64, img_cols=64, img_channels=3, nb_filters=32,
#                      pool_size=(2, 2), kernel_size=(3, 3))
#
# model.compile(optimizer=Adam(), loss={'is_two_digits': 'categorical_crossentropy', 'digit_0': 'categorical_crossentropy',
#                                       'digit_1': 'categorical_crossentropy'},
#               metrics={'is_two_digits': 'accuracy', 'digit_0': 'accuracy', 'digit_1': 'accuracy'})



json_file = 'C:/Users/HassenBELHASSEN/PycharmProjects/sn-jersey-number-annotation/train_annotations.json'
base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'
additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'

train_dataset, valid_dataset = load_dataset_soccernet_m(json_file)
print(f"train",len(train_dataset))
print(f"valid",len(valid_dataset))

print(train_dataset[0])

batch_size=4
train_batches, num_train_batches = designate_batches(train_dataset, batch_size=batch_size)
valid_batches, num_valid_batches = designate_batches(valid_dataset, batch_size=batch_size)

print(num_train_batches)
print(num_valid_batches)

train_gen = DataGenerator(train_batches, batch_size=batch_size, image_size=(64, 64))
print(train_gen.length)
valid_gen = DataGenerator(valid_batches, batch_size=batch_size, image_size=(64, 64))
print(valid_gen.length)


# # Define the DataGenerator instance
# train_gen_instance = DataGenerator(train_batches, batch_size=batch_size, image_size=(64, 64))
# valid_gen_instance = DataGenerator(valid_batches, batch_size=batch_size, image_size=(64, 64))
#
# # Create TensorFlow datasets from the generator
# train_gen = tf.data.Dataset.from_generator(
#     lambda: train_gen_instance,
#     output_signature=DataGenerator.output_signature()
# )
#
# valid_gen = tf.data.Dataset.from_generator(
#     lambda: valid_gen_instance,
#     output_signature=DataGenerator.output_signature()
# )

# history = train_model_generator(model=model,train_gen=train_gen,valid_gen=valid_gen
#                                 ,training_steps=train_gen.length,validation_steps=valid_gen.length,
#                                 checkpoint_filepath='C:/Users/HassenBELHASSEN/Desktop/models/weights/model_lost_3.keras',
#                                 epochs=10,batch_size=batch_size)


