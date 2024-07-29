from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import os
import numpy as np

# Set up the data augmentation generator with suggested parameters
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Directory containing the images
img_dir = 'C:/Users/HassenBELHASSEN/Desktop/nfl_train'
augmented_img_dir = 'C:/Users/HassenBELHASSEN/Desktop/train'

# Create a directory to save augmented images if it doesn't exist
os.makedirs(augmented_img_dir, exist_ok=True)

# Number of augmentations per image
num_augmented_images = 6  # Adjust based on your needs

# Process each image in the directory
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    img = load_img(img_path)  # Load the image
    x = img_to_array(img)  # Convert the image to a numpy array
    x = np.expand_dims(x, axis=0)  # Reshape the array

    # Generate augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        # Get the augmented image
        aug_img = array_to_img(batch[0], scale=True)

        # Save the augmented image with a new name
        aug_img_name = f"aug_{i}_{img_name}"
        aug_img.save(os.path.join(augmented_img_dir, aug_img_name))

        i += 1
        if i >= num_augmented_images:
            break  # Stop after generating the desired number of augmentations
