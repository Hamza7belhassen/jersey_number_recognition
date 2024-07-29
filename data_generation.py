import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import json
from collections import defaultdict
from random import shuffle
def chunkify(big_list, chunk_size):
    chunks = [big_list[x:x + chunk_size] for x in range(0, len(big_list), chunk_size)]
    return chunks

def load_dataset(csv_path, images_path):
    df = pd.read_csv(csv_path)
    dataset = []
    for index, row in df.iterrows():
        image_path = os.path.join(images_path, row['filename'])
        label = row[1:].values.astype(float)  # Skip the filename column and get the rest as the label
        dataset.append({"x": image_path, "y": label})
        # print('label',label.shape,label.size)
        if str(image_path)=='' :
            print('empty string')
        if label.size != 101 :
            print("empty label")
            print((label))


    return dataset

def designate_batches(dataset, batch_size=4):
    chunked_dataset = chunkify(dataset, batch_size)
    chunked_dataset = [ch for ch in chunked_dataset if len(ch)==batch_size]
    return chunked_dataset,len(chunked_dataset)

class DataGenerator(Sequence):
    def __init__(self, data_batch, batch_size=32, image_size=(128, 128)):
        self.data_batch = data_batch
        self.batch_size = batch_size
        self.image_size = image_size
        self.length = len(data_batch)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(f"{idx}/{self.length}")
        X_batch = []
        y_batch = []
        for i in range(len(self.data_batch[idx])):
            x_path = self.data_batch[idx][i]["x"]
            y_label = self.data_batch[idx][i]["y"]

            x = cv2.imread(x_path, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, self.image_size)
            x = x / 255.0

            X_batch.append(x)
            y_batch.append(y_label)

        X_batch_array = np.array(X_batch)
        y_batch_array = np.array(y_batch)

        X_batch_array = X_batch_array.astype(np.float32)
        y_batch_array = y_batch_array.astype(np.float32)


        # print('X_batch_array',X_batch_array.shape)
        # print('y_batch_array',y_batch_array.shape)

        return X_batch_array, y_batch_array



class DataLoader:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        return image.astype(np.float32)

    def preprocess_label(self, label):
        # return label.astype(np.float32)
        return np.float32(label) # for integers

    def preprocess_dataset(self, dataset):
        preprocessed_data = []
        for data in dataset:
            image_path = data['x']
            label = data['y']
            image = self.preprocess_image(image_path)
            label = self.preprocess_label(label)
            preprocessed_data.append({'x': image, 'y': label})
        return preprocessed_data


def load_dataset_csv(csv_path, images_path):
    df = pd.read_csv(csv_path)
    dataset = []
    for index, row in df.iterrows():
        image_path = os.path.join(images_path, row['filepath'])  # Use 'filepath' column for the image path
        label = row['label']  # Use 'label' column for the label
        dataset.append({"x": image_path, "y": label})

        if str(image_path) == '':
            print('empty string')
        if pd.isna(label) or label == '':  # Check for empty or NaN label
            print("empty label")
            print(label)

    return dataset


def split_dataset(dataset):
    train_dataset = dataset[0:7000]
    val_dataset = dataset[15000:17000]
    return train_dataset, val_dataset


import pandas as pd
import os
import re


def load_dataset_d(images_path):
    dataset = []
    for img_name in os.listdir(images_path):
        image_path = os.path.join(images_path, img_name)
        # Extract the class from the filename
        match = re.search(r'[HV](\d+)\.\w+$', img_name)
        if match:
            class_label = match.group(1)
            label = int(class_label)
        else:
            print(f"Could not extract class from filename: {img_name}")
            continue

        dataset.append({"x": image_path, "y": label})

        if str(image_path) == '':
            print('empty string')
        if label is None:
            print("empty label")
            print(label)

    return dataset


def read_json(file_path):
    try:
        f = open(file_path)
        data = json.load(f)
        f.close()
    except Exception as e:
        print(e)
        data = []
    return data


def load_dataset_soccernet(json_file):
    data = read_json(json_file)
    dataset = []
    base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'
    for item in data:
        image_path = item['relative_image_path']
        class_label = item['class']
        full_image_path = os.path.join(base_path, image_path.strip('/'))
        if class_label != -1:
            dataset.append({"x": full_image_path, "y": class_label})
    additional_images_dir='C:/Users/HassenBELHASSEN/Desktop/only_m1'
    if os.path.isdir(additional_images_dir):
        for image_name in os.listdir(additional_images_dir):
            image_path = os.path.join(additional_images_dir, image_name)
            # Ensure it's a file
            if os.path.isfile(image_path):
                dataset.append({"x": image_path, "y": -1})

    return dataset


def load_data_soccernet2(json_file, img_rows=64, img_cols=64, img_channels=3):
    data = read_json(json_file)
    base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'
    images = []
    is_two_digits = []
    digit_0_labels = []
    digit_1_labels = []

    for item in data:
        image_path = item['relative_image_path']
        class_label = item['class']
        full_image_path = os.path.join(base_path, image_path.strip('/'))

        # Load and preprocess the image
        if class_label != -1 :
            img = cv2.imread(full_image_path)
            img = cv2.resize(img, (img_rows, img_cols))
            images.append(img)

            # Determine if the image contains one or two digits
            if (class_label // 10) == 0 :
                is_two_digits.append(0)  # one digit
                digit_0 = class_label  # find the index of the one-hot label
                digit_0_labels.append(digit_0)
                digit_1_labels.append(10)  # special class for "no digit"
            elif (class_label // 10) != 0:
                is_two_digits.append(1)  # two digits
                digit_0_labels.append((class_label // 10))
                digit_1_labels.append((class_label % 10))


        additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'
        if os.path.isdir(additional_images_dir):
            for subdir in os.listdir(additional_images_dir):
                subdir_path = os.path.join(additional_images_dir, subdir)
                if os.path.isdir(subdir_path):
                    for image_name in os.listdir(subdir_path):
                        image_path = os.path.join(subdir_path, image_name)
                        if os.path.isfile(image_path):
                            img = cv2.imread(image_path)
                            if img is not None:
                                img = cv2.resize(img, (img_rows, img_cols))
                                images.append(img)

                                is_two_digits.append(-1)
                                digit_0_labels.append(10)
                                digit_1_labels.append(10)

    X = np.array(images, dtype='float32') / 255.0  # normalize images
    y_is_two_digits = np.array(is_two_digits, dtype='float32')
    y_digit_0 = to_categorical(digit_0_labels, num_classes=11)  # 11 classes for the first digit
    y_digit_1 = to_categorical(digit_1_labels, num_classes=11)  # 11 classes for the second digit

    return X, [y_is_two_digits, y_digit_0, y_digit_1]


def load_data_soccernet22(json_file, img_rows=64, img_cols=64, img_channels=3):
    data = read_json(json_file)
    base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'

    images = []
    is_two_digits = []
    digit_0_labels = []
    digit_1_labels = []

    grouped_data = defaultdict(list)
    for item in data:
        if item['class'] != -1:
            dir_name = item['relative_image_path'].split('/')[4]
            grouped_data[dir_name].append(item)

    processed_data = []
    for dir_name, items in grouped_data.items():
        shuffle(items)
        for item in items:
            full_image_path = os.path.join(base_path, item['relative_image_path'].strip('/'))
            img = cv2.imread(full_image_path)
            img = cv2.resize(img, (img_rows, img_cols))
            processed_data.append((img, item['class']))

    num_negative_images = int(len(processed_data) * 0.2)

    additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'
    negative_images = []
    if os.path.isdir(additional_images_dir):
        for subdir in os.listdir(additional_images_dir):
            subdir_path = os.path.join(additional_images_dir, subdir)
            if os.path.isdir(subdir_path):
                for image_name in os.listdir(subdir_path):
                    image_path = os.path.join(subdir_path, image_name)
                    if os.path.isfile(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            img = cv2.resize(img, (img_rows, img_cols))
                            negative_images.append((img, -1))

    shuffle(negative_images)
    negative_images = negative_images[:num_negative_images]

    all_data = processed_data + negative_images
    shuffle(all_data)

    for img, class_label in all_data:
        images.append(img)
        if class_label != -1:
            if class_label // 10 == 0:
                is_two_digits.append(0)
                digit_0_labels.append(class_label)
                digit_1_labels.append(10)  # special class for "no second digit"
            else:
                is_two_digits.append(1)
                digit_0_labels.append(class_label // 10)
                digit_1_labels.append(class_label % 10)
        else:
            is_two_digits.append(-1)
            digit_0_labels.append(10)
            digit_1_labels.append(10)

    X = np.array(images, dtype='float32') / 255.0
    y_is_two_digits = np.array(is_two_digits, dtype='float32')
    y_digit_0 = to_categorical(digit_0_labels, num_classes=11)
    y_digit_1 = to_categorical(digit_1_labels, num_classes=11)

    return X, [y_is_two_digits, y_digit_0, y_digit_1]



class SoccerNetDataGenerator(Sequence):
    def __init__(self, data, start_index, end_index, batch_size=32, img_rows=64, img_cols=64, img_channels=3, base_path='', additional_images_dir='', shuffle=True):
        self.data = data[start_index:end_index]
        self.batch_size = batch_size
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.base_path = base_path
        self.additional_images_dir = additional_images_dir
        self.shuffle = shuffle
        self.processed_data, self.negative_images = self._prepare_data()
        self.indexes = np.arange(len(self.processed_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.processed_data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.processed_data[k] for k in batch_indexes]
        return self._generate_X_y(batch_data)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prepare_data(self):
        grouped_data = defaultdict(list)
        for item in self.data:
            if item['class'] != -1:
                dir_name = item['relative_image_path'].split('/')[4]
                grouped_data[dir_name].append(item)

        processed_data = []
        for dir_name, items in grouped_data.items():
            shuffle(items)
            for item in items:
                full_image_path = os.path.join(self.base_path, item['relative_image_path'].strip('/'))
                img = cv2.imread(full_image_path)
                img = cv2.resize(img, (self.img_rows, self.img_cols))
                processed_data.append((img, item['class'], dir_name))

        num_negative_images = int(len(processed_data) * 0.2)
        negative_images = []
        if os.path.isdir(self.additional_images_dir):
            for subdir in os.listdir(self.additional_images_dir):
                subdir_path = os.path.join(self.additional_images_dir, subdir)
                if os.path.isdir(subdir_path):
                    for image_name in os.listdir(subdir_path):
                        image_path = os.path.join(subdir_path, image_name)
                        if os.path.isfile(image_path):
                            img = cv2.imread(image_path)
                            if img is not None:
                                img = cv2.resize(img, (self.img_rows, self.img_cols))
                                negative_images.append((img, -1, subdir))
        shuffle(negative_images)
        negative_images = negative_images[:num_negative_images]

        return processed_data, negative_images

    def _generate_X_y(self, batch_data):
        images = []
        is_two_digits = []
        digit_0_labels = []
        digit_1_labels = []

        for img, class_label, dir_name in batch_data:
            images.append(img)

            if class_label != -1:
                if class_label // 10 == 0:
                    is_two_digits.append(0)
                    digit_0_labels.append(class_label)
                    digit_1_labels.append(10)
                else:
                    is_two_digits.append(1)
                    digit_0_labels.append(class_label // 10)
                    digit_1_labels.append(class_label % 10)
            else:
                is_two_digits.append(-1)
                digit_0_labels.append(10)
                digit_1_labels.append(10)

        X = np.array(images, dtype='float32') / 255.0
        y_is_two_digits = to_categorical(is_two_digits, num_classes=3)
        y_digit_0 = to_categorical(digit_0_labels, num_classes=11)
        y_digit_1 = to_categorical(digit_1_labels, num_classes=11)

        return X, [y_is_two_digits, y_digit_0, y_digit_1]

class DataGeneratorS(Sequence):
    def __init__(self, data, batch_size=32, image_size=(64, 64)):
        self.data = data
        self.batch_size = batch_size
        self.image_size = image_size
        self.length = len(data) // batch_size
        self.indexes = np.arange(len(data))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        y_batch = [[], [], []]

        for i in indexes:
            x, y = self.data[i]
            x = cv2.resize(x, self.image_size)
            x = x / 255.0
            X_batch.append(x)
            y_batch[0].append(y[0])
            y_batch[1].append(y[1])
            y_batch[2].append(y[2])

        X_batch_array = np.array(X_batch)
        y_batch_array = [np.array(y_batch[0]), np.array(y_batch[1]), np.array(y_batch[2])]

        return X_batch_array, y_batch_array

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)



