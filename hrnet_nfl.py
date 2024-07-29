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

from hrnet import HRNetCustom



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
    img = tf.image.resize(img, [image_size, image_size])  # Resize image to the desired size
    img = img / 255.0  # Normalize image to [0, 1] range
    return img, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_data = training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
validation_data = validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

training_data_batches = training_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
validation_data_batches = validation_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

input_shape = (64, 64, 3)  # Example input shape
batch_size = 64  # Example batch size
seed_filter_size = 8  # Example seed filter size
final_filter = 100  # Number of classes
final_size = 512  # Example final size

ex = HRNetCustom(None, input_shape,batch_size, seed_filter_size, final_filter, final_size)
print(ex.model.summary())


ex.model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if labels are not one-hot encoded
    metrics=['accuracy']
)


model_checkpoint_callback = ModelCheckpoint(
filepath="C:/Users/HassenBELHASSEN/Desktop/models/weights/model_hernet_nfl.keras",
save_weights_only=False,
monitor='val_loss',
mode='min',
save_best_only=True)

callbacks = [
    model_checkpoint_callback,
    # EarlyStopping(patience=2),
    # ReduceLROnPlateau(factor=0.2, patience=3),
    # TensorBoard(log_dir='C:/Users/HassenBELHASSEN/Desktop/models/weights')
]

history = ex.model.fit(
    training_data_batches,
    validation_data=validation_data_batches,
    epochs=50,  # Number of epochs
    batch_size=batch_size,
    callbacks=callbacks
)
