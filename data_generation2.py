import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from collections import defaultdict
from random import shuffle


def chunkify(big_list, chunk_size):
    chunks = [big_list[x:x + chunk_size] for x in range(0, len(big_list), chunk_size)]
    return chunks


def read_json(json_file):
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def load_dataset_soccernet_m(json_file, validation_split=0.2):
    data = read_json(json_file)
    base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'

    dataset = []

    # Group images by directory name (numeric part)
    grouped_data = defaultdict(list)
    for item in data:
        if item['class'] != -1:
            dir_name = item['relative_image_path'].split('/')[3]
            grouped_data[dir_name].append(item)

    # Process images with class != -1
    processed_data = []
    for dir_name, items in grouped_data.items():
        shuffle(items)  # Shuffle items within each directory
        for item in items:
            full_image_path = base_path + item["relative_image_path"]
            # full_image_path = os.path.join(base_path, item['relative_image_path']
            if os.path.isfile(full_image_path):
                processed_data.append({"x": full_image_path, "y": item['class'], "dir": dir_name})

    # Determine the number of -1 class images to include
    num_negative_images = int(len(processed_data) * 0.2)

    # Process additional images with class -1
    additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'
    negative_images = []
    if os.path.isdir(additional_images_dir):
        for subdir in os.listdir(additional_images_dir):
            subdir_path = os.path.join(additional_images_dir, subdir)
            if os.path.isdir(subdir_path):
                for image_name in os.listdir(subdir_path):
                    image_path = os.path.join(subdir_path, image_name)
                    if os.path.isfile(image_path):
                        negative_images.append({"x": image_path, "y": -1, "dir": subdir})
    shuffle(negative_images)  # Shuffle the negative images
    negative_images = negative_images[:num_negative_images]

    # Combine and shuffle processed data and negative images
    all_data = processed_data + negative_images
    shuffle(all_data)

    # Interleave images by directory
    interleaved_data = []
    dir_pointers = defaultdict(list)
    for item in all_data:
        dir_pointers[item['dir']].append(item)

    while any(dir_pointers.values()):
        for dir_name in list(dir_pointers.keys()):
            if dir_pointers[dir_name]:
                interleaved_data.append(dir_pointers[dir_name].pop(0))
            if not dir_pointers[dir_name]:
                del dir_pointers[dir_name]

    # Final dataset processing
    dataset.extend(interleaved_data)

    shuffle(dataset)
    dataset = dataset[:2000]
    # Split dataset into training and validation sets
    split_index = int(len(dataset) * (1 - validation_split))
    train_dataset = dataset[:split_index]
    valid_dataset = dataset[split_index:]

    return train_dataset, valid_dataset


def designate_batches(dataset, batch_size=4):
    chunked_dataset = chunkify(dataset, batch_size)
    chunked_dataset = [ch for ch in chunked_dataset if len(ch) == batch_size]
    return chunked_dataset, len(chunked_dataset)


class DataGenerator(Sequence):
    def __init__(self, data_batch, batch_size=32, image_size=(64, 64), img_channels=3):
        self.data_batch = data_batch
        self.batch_size = batch_size
        self.image_size = image_size
        self.img_channels = img_channels
        self.length = len(data_batch)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X_batch = []
        y_is_two_digits_batch = []
        y_digit_0_batch = []
        y_digit_1_batch = []
        print(idx,self.length)

        if idx >= self.length:
            idx = idx % self.length

        for i in range(len(self.data_batch[idx])):
            # print(i,len(self.data_batch[idx]),self.length)
            x_path = self.data_batch[idx][i]["x"]
            y_class = self.data_batch[idx][i]["y"]
            # print(x_path,y_class)
            x = cv2.imread(x_path)
            x = cv2.resize(x, self.image_size)
            x = x / 255.0

            if y_class != -1:
                if y_class // 10 == 0:
                    y_is_two_digits = 0  # one digit
                    y_digit_0 = y_class
                    y_digit_1 = 10  # special class for "no digit"
                else:
                    y_is_two_digits = 1  # two digits
                    y_digit_0 = y_class // 10
                    y_digit_1 = y_class % 10
            else:
                y_is_two_digits = -1
                y_digit_0 = 10
                y_digit_1 = 10

            X_batch.append(x)
            y_is_two_digits_batch.append(y_is_two_digits)
            y_digit_0_batch.append(y_digit_0)
            y_digit_1_batch.append(y_digit_1)

        X_batch_array = np.array(X_batch, dtype='float32')
        y_is_two_digits_batch_array = to_categorical(y_is_two_digits_batch, num_classes=3)
        y_digit_0_batch_array = to_categorical(y_digit_0_batch, num_classes=11)
        y_digit_1_batch_array = to_categorical(y_digit_1_batch, num_classes=11)

        # print("X_batch_array:",X_batch_array.shape)
        # print("y_is_two_digits_batch_array:",y_is_two_digits_batch_array.shape)
        # print("y_digit_0_batch_array:",y_digit_0_batch_array.shape)
        # print("y_digit_1_batch_array:",y_digit_1_batch_array.shape)

        return X_batch_array, {"is_two_digits":y_is_two_digits_batch_array,"digit_0": y_digit_0_batch_array,"digit_1": y_digit_1_batch_array}
        # return X_batch_array , [y_is_two_digits_batch_array,y_digit_0_batch_array,y_digit_1_batch_array]



# class DataGenerator(Sequence):
#     def __init__(self, batches, batch_size=32, image_size=(64, 64), n_channels=3, shuffle=True):
#         self.batches = batches
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.n_channels = n_channels
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         return int(np.floor(len(self.batches) / self.batch_size))
#
#     def __getitem__(self, index):
#         # Generate one batch of data
#         batch = self.batches[index * self.batch_size:(index + 1) * self.batch_size]
#         X, y = self.__data_generation(batch)
#         return X, y
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.batches)
#
#     def __data_generation(self, batch):
#         # Initialize arrays
#         X = np.empty((self.batch_size, *self.image_size, self.n_channels))
#         y_is_two_digits = np.empty((self.batch_size, 3), dtype=int)
#         y_digit_0 = np.empty((self.batch_size, 11), dtype=int)
#         y_digit_1 = np.empty((self.batch_size, 11), dtype=int)
#
#         for i, (image_path, labels) in enumerate(batch):
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, self.image_size)
#             image = image / 255.0  # Normalize image
#
#             X[i,] = image
#
#             # Assume labels is a tuple (y_is_two_digits, y_digit_0, y_digit_1)
#             y_is_two_digits[i] = tf.keras.utils.to_categorical(labels[0], num_classes=3)
#             y_digit_0[i] = tf.keras.utils.to_categorical(labels[1], num_classes=11)
#             y_digit_1[i] = tf.keras.utils.to_categorical(labels[2], num_classes=11)
#
#         return X, {"is_two_digits": y_is_two_digits, "digit_0": y_digit_0, "digit_1": y_digit_1}
#
#     @staticmethod
#     def output_signature():
#         # Define the output signature
#         return (
#             tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
#             {
#                 "is_two_digits": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
#                 "digit_0": tf.TensorSpec(shape=(None, 11), dtype=tf.float32),
#                 "digit_1": tf.TensorSpec(shape=(None, 11), dtype=tf.float32)
#             }
#         )