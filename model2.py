import os

import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, Dense, BatchNormalization, Dropout, Flatten, MaxPooling2D, Input, AveragePooling2D
import tensorflow as tf


def create_model(nb_classes=11, img_rows=40, img_cols=40, img_channels=3, nb_filters=32, pool_size=(2, 2),
                 kernel_size=(3, 3)):
    inputs = Input(shape=(img_rows, img_cols, img_channels))

    # Model taken from keras example.
    cov = Conv2D(nb_filters, kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(inputs)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = Conv2D(nb_filters, kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = AveragePooling2D(pool_size=pool_size)(cov)
    cov = Dropout(0.3)(cov)

    cov = Conv2D((nb_filters * 2), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = Conv2D((nb_filters * 2), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = AveragePooling2D(pool_size=pool_size)(cov)
    cov = Dropout(0.3)(cov)

    cov = Conv2D((nb_filters * 4), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = Conv2D((nb_filters * 4), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = AveragePooling2D(pool_size=pool_size)(cov)
    cov = Dropout(0.3)(cov)

    cov = Conv2D((nb_filters * 8), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = Conv2D((nb_filters * 8), kernel_size=(kernel_size[0], kernel_size[1]), padding="same")(cov)
    cov = BatchNormalization()(cov)
    cov = Activation('relu')(cov)
    cov = AveragePooling2D(pool_size=pool_size)(cov)
    cov = Dropout(0.3)(cov)

    cov_out = Flatten()(cov)

    # DIGIT 0 NET
    cov2 = Dense(128, activation='relu')(cov_out)
    cov2 = BatchNormalization()(cov2)
    cov2 = Dropout(0.3)(cov2)
    cov2 = Dense(64, activation='relu')(cov2)
    cov2 = BatchNormalization()(cov2)
    cov2 = Dropout(0.3)(cov2)
    cov2 = Dense(64, activation='relu')(cov2)
    cov2 = BatchNormalization()(cov2)
    cov2 = Dropout(0.3)(cov2)
    c0 = Dense(nb_classes, activation='softmax', name="digit_0")(cov2)

    # DIGIT 1 NET
    cov3 = Dense(128, activation='relu')(cov_out)
    cov3 = BatchNormalization()(cov3)
    cov3 = Dropout(0.3)(cov3)
    cov3 = Dense(64, activation='relu')(cov3)
    cov3 = BatchNormalization()(cov3)
    cov3 = Dropout(0.3)(cov3)
    cov3 = Dense(64, activation='relu')(cov3)
    cov3 = BatchNormalization()(cov3)
    cov3 = Dropout(0.3)(cov3)
    c1 = Dense(nb_classes, activation='softmax', name="digit_1")(cov3)

    # Defining the model
    model = tf.keras.Model(inputs=inputs, outputs=[c0, c1])
    return model


def load_model_from_file(model_path):
    nb_classes = 11
    # image input dimensions
    img_rows = 64
    img_cols = 64
    img_channels = 3
    # number of convulation filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    model = create_model(nb_classes=nb_classes, img_rows=img_rows, img_cols=img_cols,
                         img_channels=img_channels, nb_filters=nb_filters, pool_size=pool_size,
                         kernel_size=kernel_size)
    model.load_weights(model_path)
    return model


number_model_path = ('C:/Users/HassenBELHASSEN/PycharmProjects/jersey_number/JerseyNet.h5')
number_model = load_model_from_file(number_model_path)

#number_model.summary()


import cv2
base_images_pat  = "C:/Users/HassenBELHASSEN/Desktop/alphapose_examples/1/alphapose_test_output/"
images = [cv2.resize(cv2.imread(base_images_pat+f), (64,64)) for f in os.listdir(base_images_pat)]
images_array = np.array(images)

print("images_array: ", images_array.shape)
preds = number_model(images_array)


def show_image(my_image, size=(540, 960)):
    resized_image = cv2.resize(my_image, size)
    cv2.imshow("image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# print("preds.shape: " ,preds.shape)

for img, pred in zip(images, preds[0]):
    predicted_class = np.argmax(pred)
    print('pred: ', pred.shape, pred)
    print('Predicted class: ', predicted_class)
    show_image(img)
