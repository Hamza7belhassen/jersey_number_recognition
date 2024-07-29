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
from data_generation import SoccerNetDataGenerator , read_json , load_data_soccernet22 , DataGeneratorS
from data_generation2 import load_dataset_soccernet_m , designate_batches ,DataGenerator



def load_data(data_path, img_rows=64, img_cols=64, img_channels=3):
    images = []
    is_two_digits = []
    digit_0_labels = []
    digit_1_labels = []

    labels_csv = os.path.join(data_path, '_classes.csv')
    images_folder = data_path

    df = pd.read_csv(labels_csv)

    for index, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(images_folder, filename)

        # Load and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_rows, img_cols))
        images.append(img)

        # Determine if the image contains one or two digits
        digit_sum = sum(row[1:])  # sum of the one-hot encoded labels
        if digit_sum == 1:
            is_two_digits.append(0)  # one digit
            digit_0 = np.argmax(row[1:])  # find the index of the one-hot label
            digit_0_labels.append(digit_0)
            digit_1_labels.append(10)  # special class for "no digit"
        else:
            is_two_digits.append(1)  # two digits
            non_zero_indices = np.where(row[1:].values != 0)[0]
            digit_0_labels.append(non_zero_indices[0])
            digit_1_labels.append(non_zero_indices[1])

    X = np.array(images, dtype='float32') / 255.0  # normalize images
    y_is_two_digits = np.array(is_two_digits, dtype='float32')
    y_digit_0 = to_categorical(digit_0_labels, num_classes=10)  # 10 classes for the first digit
    y_digit_1 = to_categorical(digit_1_labels, num_classes=11)  # 11 classes for the second digit

    return X, [y_is_two_digits, y_digit_0, y_digit_1]



def load_data_from_multiple_paths(base_path, img_rows=64, img_cols=64, img_channels=3, test_size=0.2, random_state=42):
    images = []
    is_two_digits = []
    digit_0_labels = []
    digit_1_labels = []

    # Loop through each subdirectory in the base path
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            labels_csv = os.path.join(subdir_path, 'new_classes.csv')

            if not os.path.exists(labels_csv):
                continue

            df = pd.read_csv(labels_csv)

            for index, row in df.iterrows():
                filename = row['filename']
                img_path = os.path.join(subdir_path, filename)

                digit_sum = sum(row[1:])  # sum of the one-hot encoded labels for digits 0-9
                if digit_sum == 1:
                    # Load and preprocess the image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue  # Skip if the image cannot be loaded
                    img = cv2.resize(img, (img_rows, img_cols))
                    images.append(img)

                    is_two_digits.append(0)  # one digit
                    digit_0 = np.argmax(row[1:])  # find the index of the one-hot label
                    digit_0_labels.append(digit_0)
                    digit_1_labels.append(10)  # special class for "no digit"
                elif digit_sum == 2:
                    non_zero_indices = np.where(row[1:].values != 0)[0]
                    if len(non_zero_indices) < 2:
                        continue  # Skip adding the image if there are not enough non-zero indices

                    # Load and preprocess the image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue  # Skip if the image cannot be loaded
                    img = cv2.resize(img, (img_rows, img_cols))
                    images.append(img)

                    is_two_digits.append(1)  # two digits
                    digit_0_labels.append(non_zero_indices[0])
                    digit_1_labels.append(non_zero_indices[1])

    X = np.array(images, dtype='float32') / 255.0  # normalize images
    y_is_two_digits = np.array(is_two_digits, dtype='float32')
    y_digit_0 = to_categorical(digit_0_labels, num_classes=10)  # 10 classes for the first digit
    y_digit_1 = to_categorical(digit_1_labels, num_classes=11)  # 11 classes for the second digit (10 digits + 1 "no digit" class)

    # Shuffle the data
    X, y_is_two_digits, y_digit_0, y_digit_1 = shuffle(X, y_is_two_digits, y_digit_0, y_digit_1, random_state=random_state)

    # Split the data into training and validation sets
    X_train, X_valid, y_is_two_digits_train, y_is_two_digits_valid = train_test_split(X, y_is_two_digits, test_size=test_size, random_state=random_state)
    y_digit_0_train, y_digit_0_valid = train_test_split(y_digit_0, test_size=test_size, random_state=random_state)
    y_digit_1_train, y_digit_1_valid = train_test_split(y_digit_1, test_size=test_size, random_state=random_state)

    return (X_train, [y_is_two_digits_train, y_digit_0_train, y_digit_1_train]), (X_valid, [y_is_two_digits_valid, y_digit_0_valid, y_digit_1_valid])

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
    model = Model(inputs=inputs, outputs=[is_two_digits, c0, c1])
    return model

#
# # Load the training and validation data
# train_path = 'C:/Users/HassenBELHASSEN/Desktop/number.v2i.multiclass/train'
# valid_path = 'C:/Users/HassenBELHASSEN/Desktop/number.v2i.multiclass/valid'
#
# X_train, y_train = load_data(train_path)
# X_valid, y_valid = load_data(valid_path)

# base_path = 'C:/Users/HassenBELHASSEN/Desktop/hamza'
# (train_data, train_labels), (valid_data, valid_labels) = load_data_from_multiple_paths(base_path)


json_file = 'C:/Users/HassenBELHASSEN/PycharmProjects/sn-jersey-number-annotation/train_annotations.json'
base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'
additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'



train_dataset, valid_dataset = load_dataset(json_file)

train_batches, num_train_batches = designate_batches(train_dataset, batch_size=32)
valid_batches, num_valid_batches = designate_batches(valid_dataset, batch_size=32)

train_gen = DataGenerator(train_batches, batch_size=32, image_size=(64, 64))
valid_gen = DataGenerator(valid_batches, batch_size=32, image_size=(64, 64)


# data = read_json(json_file)
#
# train_start_index = 0
# train_end_index = 600000
# val_start_index = 600000
# val_end_index = len(data)
#
# train_data_generator = SoccerNetDataGenerator(data, train_start_index, train_end_index, batch_size=32, base_path=base_path, additional_images_dir=additional_images_dir, shuffle=True)
# val_data_generator = SoccerNetDataGenerator(data, val_start_index, val_end_index, batch_size=32, base_path=base_path, additional_images_dir=additional_images_dir, shuffle=False)
#
#
# def data_generator_to_tf_dataset(generator, output_signature):
#     def generator_fn():
#         for batch in generator:
#             yield batch
#     return tf.data.Dataset.from_generator(generator_fn, output_signature=output_signature)
#
#
# # Define the output signature
# output_signature = (
#     tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
#     (
#         tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, 11), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, 11), dtype=tf.float32),
#     )
# )
#
#
# # Convert the data generators to tf.data.Dataset
# train_dataset = data_generator_to_tf_dataset(train_data_generator, output_signature)
# val_dataset = data_generator_to_tf_dataset(val_data_generator, output_signature)
#
# # Prefetch data to improve performance
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# Create the model
model = create_model(nb_classes_0=11, nb_classes_1=11, img_rows=64, img_cols=64, img_channels=3, nb_filters=32,
                     pool_size=(2, 2), kernel_size=(3, 3))

# Compile the model
model.compile(optimizer=Adam(), loss={'is_two_digits': 'binary_crossentropy', 'digit_0': 'categorical_crossentropy',
                                      'digit_1': 'categorical_crossentropy'},
              metrics={'is_two_digits': 'accuracy', 'digit_0': 'accuracy', 'digit_1': 'accuracy'})


model_checkpoint_callback = ModelCheckpoint(
filepath="C:/Users/HassenBELHASSEN/Desktop/models/weights/model_lost_3.keras",
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

# Train the model
model.fit(train_generator,validation_data=val_generator, epochs=50,callbacks=callbacks)


