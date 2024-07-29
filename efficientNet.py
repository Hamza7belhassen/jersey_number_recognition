import sys
from pathlib import Path
import os
from datetime import datetime
import time
import random
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomContrast



dataset_df = pd.read_csv("C:/Users/HassenBELHASSEN/Desktop/archive/train_player_numbers.csv")


dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
dataset_df["filepath"] = ["C:/Users/HassenBELHASSEN/Desktop/archive/"+row.filepath for idx, row in dataset_df.iterrows()]
dataset_df = dataset_df[dataset_df.video_frame.str.contains("Endzone")]

training_percentage = 0.8
training_item_count = int(len(dataset_df)*training_percentage)
validation_item_count = len(dataset_df)-int(len(dataset_df)*training_percentage)
training_df = dataset_df[:training_item_count]
validation_df = dataset_df[training_item_count:]


batch_size = 64
image_size = 64
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
classes_to_predict = sorted(training_df.label.unique())
class_weights = compute_class_weight("balanced", classes=classes_to_predict, y=training_df.label)
class_weights_dict = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}

training_data = tf.data.Dataset.from_tensor_slices((training_df.filepath.values, training_df.label.values))
validation_data = tf.data.Dataset.from_tensor_slices((validation_df.filepath.values, validation_df.label.values))


def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_data = training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
validation_data = validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

training_data_batches = training_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
validation_data_batches = validation_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)


data_augmentation_layers = tf.keras.Sequential(
    [
        RandomRotation(0.25),
        RandomZoom((-0.2, 0)),
        RandomContrast((0.2,0.2))
    ]
)

# image = Image.open(training_df.filepath.values[1])
# plt.imshow(image)
# plt.show()
#
# image = tf.expand_dims(np.array(image), 0)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#   augmented_image = data_augmentation_layers(image)
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(augmented_image[0].numpy().astype("uint8"))
#   plt.axis("off")
#
# plt.show()

efficientnet = EfficientNetB0(weights="C:/Users/HassenBELHASSEN/PycharmProjects/jersey_number/efficientnetb0_notop.h5",
                              include_top=False,
                              input_shape=input_shape)

inputs = Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
efficientnet = efficientnet(augmented)
pooling = layers.GlobalAveragePooling2D()(efficientnet)
dropout = layers.Dropout(dropout_rate)(pooling)
outputs = Dense(len(classes_to_predict), activation="softmax")(dropout)
model = Model(inputs=inputs, outputs=outputs)

print(model.summary())

epochs = 50
decay_steps = int(round(len(training_df)/batch_size))*epochs
cosine_decay = CosineDecay(initial_learning_rate=1e-3, decay_steps=decay_steps, alpha=0.3)

callbacks = [ModelCheckpoint(filepath='C:/Users/HassenBELHASSEN/Desktop/models/weights/model_efficentnet.keras', monitor='val_loss', save_best_only=True)]

model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(cosine_decay), metrics=["accuracy"])

history = model.fit(training_data_batches,
                  epochs = epochs,
                  validation_data=validation_data_batches,
                  class_weight=class_weights_dict,
                  callbacks=callbacks)