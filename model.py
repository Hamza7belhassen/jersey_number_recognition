from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from data_generation import  designate_batches,DataGenerator,load_dataset,DataLoader
from train import train_model_generator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from data_generation import load_dataset_csv,split_dataset,DataLoader

def create_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Print model summary
    print(model.summary())

    return model


train_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/train')
test_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/test/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/test')
valid_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/valid')

#
# batch_size=4
# train_batches,train_steps=designate_batches(train_dataset,batch_size)
# test_batches,test_steps=designate_batches(test_dataset,batch_size)
# valid_batches,valid_steps=designate_batches(valid_dataset,batch_size)
#
# train_dt=DataGenerator(train_batches,batch_size)
# valid_dt=DataGenerator(valid_batches,batch_size)

# train_model_generator(model,train_dt,valid_dt,train_steps,checkpoint_filepath="C:/Users/HassenBELHASSEN/Desktop/models/model_checkpoint.keras",epochs=50,batch_size=batch_size,validation_steps=valid_steps)


loader = DataLoader(image_size=(128,128))
train_dataset = loader.preprocess_dataset(train_dataset)
valid_dataset = loader.preprocess_dataset(valid_dataset)
test_dataset = loader.preprocess_dataset(test_dataset)

train_images = np.array([data['x'] for data in train_dataset])
train_labels = np.array([data['y'] for data in train_dataset])

valid_images = np.array([data['x'] for data in valid_dataset])
valid_labels = np.array([data['y'] for data in valid_dataset])

test_images = np.array([data['x'] for data in test_dataset])
test_labels = np.array([data['y'] for data in test_dataset])



# csv_path = 'C:/Users/HassenBELHASSEN/Desktop/archive/endzone_images.csv'
# images_path = 'C:/Users/HassenBELHASSEN/Desktop/archive/endzone_images'
# dataset = load_dataset_csv(csv_path, images_path)
#
# train_dataset, val_dataset = split_dataset(dataset)
#
# # Preprocess datasets using your DataLoader
# loader = DataLoader(image_size=(64, 64))
# train_dataset = loader.preprocess_dataset(train_dataset)
# val_dataset = loader.preprocess_dataset(val_dataset)
#
# train_images = np.array([data['x'] for data in train_dataset])
# train_labels = np.array([data['y'] for data in train_dataset])
#
# val_images = np.array([data['x'] for data in val_dataset])
# val_labels = np.array([data['y'] for data in val_dataset])


input_shape=(128,128,3)

model =create_model(input_shape,100)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint_callback = ModelCheckpoint(
filepath="C:/Users/HassenBELHASSEN/Desktop/models/weights/test_model.keras",
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

history = model.fit(train_images, train_labels,epochs=50,batch_size=4,validation_data=(valid_images, valid_labels),callbacks=callbacks)
