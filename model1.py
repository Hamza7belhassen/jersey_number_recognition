from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from data_generation import load_dataset_d, DataLoader
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

# Load your datasets
train_data = load_dataset_d('C:/Users/HassenBELHASSEN/Desktop/train')
valid_data = load_dataset_d('C:/Users/HassenBELHASSEN/Desktop/valid')

loader = DataLoader(image_size=(128, 128))
train_dataset = loader.preprocess_dataset(train_data)
val_dataset = loader.preprocess_dataset(valid_data)

train_images = np.array([data['x'] for data in train_dataset])
train_labels = np.array([data['y'] for data in train_dataset])

val_images = np.array([data['x'] for data in val_dataset])
val_labels = np.array([data['y'] for data in val_dataset])

# Load the previously trained model
pretrained_model = load_model('C:/Users/HassenBELHASSEN/Desktop/models/weights/model_hernet_final_inchalah.keras')

# Extract output from last layer of pretrained model
pretrained_output = pretrained_model.layers[-2].output  # Output of the last Dense layer before the final prediction layer

# Freeze layers of pretrained model
for layer in pretrained_model.layers:
    layer.trainable = False

# Create new model and add layers
new_model = Sequential()
new_model.add(pretrained_model.layers[0])  # Add layers up to last layer before final prediction
new_model.add(Flatten())  # Flatten the output from the pretrained model
new_model.add(Dense(units=256, activation='relu'))  # Add new dense layers
new_model.add(Dense(units=101, activation='softmax'))  # Adjust units based on your number of classes

# Compile the new model
new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Define callbacks, like ModelCheckpoint
model_checkpoint_callback = ModelCheckpoint(
    filepath="C:/Users/HassenBELHASSEN/Desktop/models/weights/model_comp.keras",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

callbacks = [model_checkpoint_callback]

# Train the new model
batch_size = 32
epochs = 50

history = new_model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks
)
