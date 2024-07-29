import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import UpSampling2D, add, concatenate
import tensorflow as tf
import numpy as np
import warnings
from data_generation import  load_dataset,DataLoader,load_dataset_csv,split_dataset,load_dataset_d
from tensorflow.keras.callbacks import ModelCheckpoint

# Suppress all warnings
warnings.filterwarnings("ignore")

from keras.models import load_model


class HRNetCustom:
    def __init__(self, model_path=None, input_shape=(128, 128, 3), batch_size=2, seed_filter_size=32, final_filter=91,
                 final_size='same'):
        self.input_shape = input_shape
        self.final_filter = final_filter
        self.batch_size = batch_size
        self.seed_filter_size = seed_filter_size
        self.final_size = final_size
        if model_path is not None:
            self.model = load_model(model_path,
                                    custom_objects={'weighted_binary_crossentropy': self.weighted_binary_crossentropy,
                                                    'iou': self.iou, 'dice_loss': self.dice_loss,
                                                    'combo_loss': self.combo_loss})
        else:
            self.model = self.build_model()

    @staticmethod
    def conv3x3(x, out_filters, strides=(1, 1)):
        x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
        return x

    def basic_Block(self, input, out_filters, strides=(1, 1), with_conv_shortcut=False):
        x = self.conv3x3(input, out_filters, strides)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = self.conv3x3(x, out_filters)
        x = BatchNormalization(axis=3)(x)

        if with_conv_shortcut:
            residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
            residual = BatchNormalization(axis=3)(residual)
            x = add([x, residual])
        else:
            x = add([x, input])

        x = Activation('relu')(x)
        return x

    @staticmethod
    def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
        expansion = 4
        de_filters = int(out_filters / expansion)

        x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)

        if with_conv_shortcut:
            residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
            residual = BatchNormalization(axis=3)(residual)
            x = add([x, residual])
        else:
            x = add([x, input])

        x = Activation('relu')(x)
        return x

    def stem_net(self, input):
        x = Conv2D(2 * self.seed_filter_size, 3, strides=(2, 2), padding='same', use_bias=False,
                   kernel_initializer='he_normal')(input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        # x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        # x = BatchNormalization(axis=3)(x)
        # x = Activation('relu')(x)

        x = self.bottleneck_Block(x, 8 * self.seed_filter_size, with_conv_shortcut=True)
        x = self.bottleneck_Block(x, 8 * self.seed_filter_size, with_conv_shortcut=False)
        x = self.bottleneck_Block(x, 8 * self.seed_filter_size, with_conv_shortcut=False)
        x = self.bottleneck_Block(x, 8 * self.seed_filter_size, with_conv_shortcut=False)

        return x

    @staticmethod
    def transition_layer1(x, out_filters_list=[32, 64]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)

        x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)

        return [x0, x1]

    def make_branch1_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch1_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    @staticmethod
    def fuse_layer1(x, seed_filter):
        x0_0 = x[0]
        x0_1 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0 = add([x0_0, x0_1])

        x1_0 = Conv2D(2 * seed_filter, 3, strides=(2, 2), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1 = add([x1_0, x1_1])
        return [x0, x1]

    @staticmethod
    def transition_layer2(x, out_filters_list=[32, 64, 128]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)

        x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)

        return [x0, x1, x2]

    def make_branch2_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch2_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch2_2(self, x, out_filters=128):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    @staticmethod
    def fuse_layer2(x, seed_filter=32):
        x0_0 = x[0]
        x0_1 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0 = add([x0_0, x0_1, x0_2])

        x1_0 = Conv2D(seed_filter * 2, 3, strides=(2, 2), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1_2 = Conv2D(seed_filter * 2, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x1_2 = BatchNormalization(axis=3)(x1_2)
        x1_2 = UpSampling2D(size=(2, 2))(x1_2)
        x1 = add([x1_0, x1_1, x1_2])

        x2_0 = Conv2D(seed_filter, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(
            x[0])
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_0 = Activation('relu')(x2_0)
        x2_0 = Conv2D(seed_filter * 4, 3, strides=(2, 2), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x2_0)
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_1 = Conv2D(seed_filter * 4, 3, strides=(2, 2), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x[1])
        x2_1 = BatchNormalization(axis=3)(x2_1)
        x2_2 = x[2]
        x2 = add([x2_0, x2_1, x2_2])
        return [x0, x1, x2]

    @staticmethod
    def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)

        x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)

        x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
        x3 = BatchNormalization(axis=3)(x3)
        x3 = Activation('relu')(x3)

        return [x0, x1, x2, x3]

    def make_branch3_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch3_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch3_2(self, x, out_filters=128):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    def make_branch3_3(self, x, out_filters=256):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x

    @staticmethod
    def fuse_layer3(x, seed_filter):
        x0_0 = x[0]
        x0_1 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0_3 = Conv2D(seed_filter, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
        x0_3 = BatchNormalization(axis=3)(x0_3)
        x0_3 = UpSampling2D(size=(8, 8))(x0_3)
        x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
        return x0

    @staticmethod
    def final_layer(x, classes=1, final_size='same'):
        # x = UpSampling2D(size=(2, 2))(x)
        if final_size == 'half':
            x = UpSampling2D(size=(1, 1))(x)
        elif final_size == 'same':
            x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('sigmoid', name='Classification')(x)
        return x

    @staticmethod
    def classification_layer(x, classes, final_size):
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Pooling to reduce dimensions
        x = tf.keras.layers.Dense(final_size, activation='relu')(x)  # Fully connected layer with ReLU activation
        x = tf.keras.layers.Dense(classes, activation='softmax')(
            x)  # Final classification layer with softmax activation
        return x

    # @staticmethod
    # def classification_layer(x, classes=1, final_size='same'):
    #
    #     return x


    def build_model(self):
        inputs = Input(batch_shape=(self.batch_size,) + self.input_shape)

        x = self.stem_net(inputs)
        x = self.transition_layer1(x, out_filters_list=[self.seed_filter_size, self.seed_filter_size * 2])
        x0 = self.make_branch1_0(x[0], out_filters=self.seed_filter_size)
        x1 = self.make_branch1_1(x[1], out_filters=self.seed_filter_size * 2)
        x = self.fuse_layer1([x0, x1], seed_filter=self.seed_filter_size)

        x = self.transition_layer2(x, out_filters_list=[self.seed_filter_size, self.seed_filter_size * 2,
                                                        self.seed_filter_size * 4])
        x0 = self.make_branch2_0(x[0], out_filters=self.seed_filter_size)
        x1 = self.make_branch2_1(x[1], out_filters=self.seed_filter_size * 2)
        x2 = self.make_branch2_2(x[2], out_filters=self.seed_filter_size * 4)
        x = self.fuse_layer2([x0, x1, x2], seed_filter=self.seed_filter_size)

        x = self.transition_layer3(x, out_filters_list=[self.seed_filter_size, self.seed_filter_size * 2,
                                                        self.seed_filter_size * 4, self.seed_filter_size * 8])
        x0 = self.make_branch3_0(x[0], out_filters=self.seed_filter_size)
        x1 = self.make_branch3_1(x[1], out_filters=self.seed_filter_size * 2)
        x2 = self.make_branch3_2(x[2], out_filters=self.seed_filter_size * 4)
        x3 = self.make_branch3_3(x[3], out_filters=self.seed_filter_size * 8)
        x = self.fuse_layer3([x0, x1, x2, x3], seed_filter=self.seed_filter_size)
        out = self.classification_layer(x, classes=self.final_filter, final_size=self.final_size)
        # out = self.final_layer(x, classes=self.final_filter, final_size=self.final_size)
        model = Model(inputs=inputs, outputs=out)

        return model

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coeff

    @staticmethod
    def weighted_binary_crossentropy(y_true, y_pred, positive_weight=10):
        y_pred = K.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
        weights = K.ones_like(y_pred)  # (None,560,960,91)
        # print('y_pred: ', y_pred.shape)
        # print('positive_weight: ', positive_weight)
        # print('weights: ', weights.shape)
        weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
        out = K.binary_crossentropy(y_true, y_pred)  # (None,560,960,91)
        out = out * weights  # (None,560,960,91)* (None,560,960,91)
        return K.mean(out, axis=(1, 2, 3))  # Calculate mean loss over spatial dimensions

    # @staticmethod
    # def weighted_binary_crossentropy(y_true, y_pred, positive_weight=10):
    #     y_pred = K.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
    #     # print(f"Shapes before any modifications - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    #     weights = K.ones_like(y_pred)  # Same shape as y_pred
    #     if not isinstance(positive_weight, (int, float)):
    #         positive_weight = tf.convert_to_tensor(positive_weight, dtype=y_pred.dtype)
    #         positive_weight = tf.expand_dims(positive_weight, axis=-1)
    #         print('y_pred: ', y_pred.shape)
    #         print('positive_weight: ', positive_weight.shape)
    #         print('weights: ', weights.shape)
    #         print('------------------------------------------------------------------------------------------------------')
    #         weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
    #     # print(f"Shapes after tf.where - weights: {weights.shape}, y_pred: {y_pred.shape}")
    #     out = K.binary_crossentropy(y_true, y_pred)
    #     out = out * weights
    #     # print(f"Shapes before reduction - out: {out.shape}")
    #     return K.mean(out, axis=(1, 2, 3))

    @staticmethod
    def iou(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        return intersection / (union + K.epsilon())

    def combo_loss(self, y_true, y_pred):
        try:
            wbce = self.weighted_binary_crossentropy(y_true, y_pred)
        except: wbce = 0
        dice = self.dice_loss(y_true, y_pred)
        return wbce + dice

    def compile_model(self, loss=None, learning_rate=1e-3):
        if loss == 'wbce':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                               loss=self.weighted_binary_crossentropy, metrics=[self.iou, 'accuracy'])
        elif loss == 'dice':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.dice_loss,
                               metrics=[self.iou, 'accuracy'])
        # elif loss == 'combo':
        #     self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.combo_loss,
        #                        metrics=[self.iou, 'accuracy'])
        elif loss == 'combo':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.combo_loss,
                               metrics=[self.iou, 'accuracy'])

        elif loss == 'mse':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse',
                               metrics=[self.iou, 'accuracy'])

        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                               loss='binary_crossentropy', metrics=[self.iou])

    def train_model_generator(self, train_gen, valid_gen, steps_per_epoch, checkpoint_filepath, validation_steps,
                              epochs=100, batch_size=32, verbose=1):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        def lr_schedule(epoch):
            if epoch < 50:
                return 1e-3
            elif epoch < 70:
                return 1e-4
            elif epoch < 90:
                return 1e-5
            else:
                return 1e-6

        # Create a LearningRateScheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # callbacks = [model_checkpoint_callback]
        callbacks = [model_checkpoint_callback, lr_scheduler]

        history = self.model.fit(
            train_gen,  # Training generator
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_gen,  # Validation generator
            validation_steps=validation_steps,  # Number of validation steps
            callbacks=callbacks
        )
        return history

train_data = load_dataset_d('C:/Users/HassenBELHASSEN/Desktop/train')
valid_data = load_dataset_d('C:/Users/HassenBELHASSEN/Desktop/valid')

loader = DataLoader(image_size=(64, 64))
train_dataset = loader.preprocess_dataset(train_data)
val_dataset = loader.preprocess_dataset(valid_data)

train_images = np.array([data['x'] for data in train_dataset])
train_labels = np.array([data['y'] for data in train_dataset])

val_images = np.array([data['x'] for data in val_dataset])
val_labels = np.array([data['y'] for data in val_dataset])



# train_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes_m.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/train')
# # test_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/test/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/test')
# valid_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes_m.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/valid')
#
# loader = DataLoader(image_size=(128,128))
# train_dataset = loader.preprocess_dataset(train_dataset)
# valid_dataset = loader.preprocess_dataset(valid_dataset)
# # test_dataset = loader.preprocess_dataset(test_dataset)
#
# train_images = np.array([data['x'] for data in train_dataset])
# train_labels = np.array([data['y'] for data in train_dataset])
#
# valid_images = np.array([data['x'] for data in valid_dataset])
# valid_labels = np.array([data['y'] for data in valid_dataset])

# test_images = np.array([data['x'] for data in test_dataset])
# test_labels = np.array([data['y'] for data in test_dataset])

# csv_path = 'C:/Users/HassenBELHASSEN/Desktop/archive/endzone_images.csv'
# images_path = 'C:/Users/HassenBELHASSEN/Desktop/archive/endzone_images'
# dataset = load_dataset_csv(csv_path, images_path)
#
# train_dataset, val_dataset = split_dataset(dataset)
#
# # Preprocess datasets using your DataLoader
# loader = DataLoader(image_size=(128, 128))
# train_dataset = loader.preprocess_dataset(train_dataset)
# val_dataset = loader.preprocess_dataset(val_dataset)
#
# train_images = np.array([data['x'] for data in train_dataset])
# train_labels = np.array([data['y'] for data in train_dataset])
#
# val_images = np.array([data['x'] for data in val_dataset])
# val_labels = np.array([data['y'] for data in val_dataset])



input_shape = (64, 64, 3)  # Example input shape
batch_size = 32  # Example batch size
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
filepath="C:/Users/HassenBELHASSEN/Desktop/models/weights/model_nfl.keras",
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
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=50,  # Number of epochs
    batch_size=batch_size,
    callbacks=callbacks
)