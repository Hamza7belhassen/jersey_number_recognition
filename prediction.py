import cv2
import numpy as np
import os
import tensorflow as tf
from data_generation import  load_dataset,DataLoader
from pprint import pprint
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('C:/Users/HassenBELHASSEN/Desktop/models/weights/model_lost_2.keras')


class_labels = list((-1, 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99))


#Preprocess function for new images
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Resize to match the input size of the model
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)


# def preprocess_image(image_path, image_size=64):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [image_size, image_size])
#     img = img / 255.0
#     img = tf.expand_dims(img, 0)  # Add batch dimension
#     return img


def predict_image(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# image_path = "C:/Users/HassenBELHASSEN/Desktop/hamza/train/_1VC8463_H_jpg.rf.be9698a4452473fedc002d070e71e240.jpg"
# predicted_class = predict_image(model, image_path)
# print(f"Predicted class: {predicted_class[0]}")
#
# #Directory containing the new images
# #path = 'C:/Users/HassenBELHASSEN/Desktop/alphapose_examples/1/alphapose_test_output'
# #path = 'C:/Users/HassenBELHASSEN/Desktop/archive/train_player_numbers'
# path='C:/Users/HassenBELHASSEN/Desktop/alphapose_examples/2/extracted_images/0'
# # res=[]
# for file in os.listdir(path):
#     img_path = os.path.join(path, file)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     preprocessed_img = preprocess_image(img)
#     prediction = model.predict(preprocessed_img)
    #print(prediction)

    # predicted_class = np.argmax(prediction)
    # print(f"Predicted class: {predicted_class}")
    # plt.imshow(img)
    # plt.title(f"Predicted Class: {predicted_class}")
    # plt.axis('off')
    # plt.show()
    # predicted_class_index = np.argmax(prediction)
    # predicted_class_label = class_labels[predicted_class_index]
    # predicted_probability = prediction[0][predicted_class_index]
    # text=str(predicted_class_label)+' / '+str(predicted_probability)
    # a=cv2.resize(img,(560,560))
    # cv2.putText(a,text ,(80,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=None, lineType=None, bottomLeftOrigin=None)
    # cv2.imshow('Original Image', a)
    # cv2.waitKey(0)
    # a={"image":file,"class":predicted_class_label,"prob":predicted_probability}
    # res.append(a)

# pprint(res)



# test_dataset=load_dataset('C:/Users/HassenBELHASSEN/Desktop/hamza/test/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/test')
# loader = DataLoader(image_size=(128,128))
# test_dataset = loader.preprocess_dataset(test_dataset)
#
# test_images = np.array([data['x'] for data in test_dataset])
# test_labels = np.array([data['y'] for data in test_dataset])
#
# results = model.evaluate(test_images, test_labels, verbose=1)
# print(f"Test Loss: {results[0]}")
# print(f"Test Accuracy: {results[1]}")

# img = cv2.imread('C:/Users/HassenBELHASSEN/Desktop/archive/train_player_numbers/57583_000082_Endzone_20_H27.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# preprocessed_img = preprocess_image(img)
# prediction = model.predict(preprocessed_img)
# print(prediction)

def load_test_data(test_path, img_rows=64, img_cols=64):
    images = []
    filenames = []

    for filename in os.listdir(test_path):
        img_path = os.path.join(test_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_rows, img_cols))
        images.append(img)
        filenames.append(filename)

    X_test = np.array(images, dtype='float32') / 255.0  # Normalize images

    return X_test, filenames

test_path = 'C:/Users/HassenBELHASSEN/Desktop/nfl_train'
X_test, test_filenames = load_test_data(test_path)
print(model.summary())

predictions = model.predict(X_test)

is_two_digits_pred, digit_0_pred, digit_1_pred = predictions
#
# for filename, is_two_digits, digit_0, digit_1 in zip(test_filenames, is_two_digits_pred, digit_0_pred, digit_1_pred):
#     print(f"Filename: {filename}")
#     print(f"Is two digits: {is_two_digits}")
#     print(f"Digit 0 prediction: {np.argmax(digit_0)}")
#     print(f"Digit 1 prediction: {np.argmax(digit_1) if is_two_digits > 0.5 else 'None'}")  # Only print if two digits predicted
#     print()


def write_predictions_on_image(image, is_two_digits, digit_0, digit_1):
    # Convert predictions to digits
    digit_0_pred = np.argmax(digit_0)
    digit_1_pred = np.argmax(digit_1) if is_two_digits > 0.5 else 'None'

    # Define the text to put on the image
    text = (
            f"Is two digits: {is_two_digits[0]}\n"
            f"Digit 0 prediction: {digit_0_pred}\n"
            f"Digit 1 prediction: {digit_1_pred}")

    # Split text into lines
    lines = text.split('\n')

    # Define position, font, scale, and color for the text
    position = (10, 30)  # Top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)  # Blue color in BGR
    thickness = 1
    line_height = 25
    image = cv2.resize(image , (300,300))
    # Put text on the image
    # Put text on the image
    for i, line in enumerate(lines):
        y_position = position[1] + i * line_height
        cv2.putText(image, line, (position[0], y_position), font, font_scale, color, thickness, cv2.LINE_AA)

    return image
    # cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    #
    # return image


def concatenate_images_vertically(images):
    if len(set(image.shape[1] for image in images)) != 1:
        raise ValueError("Images must have the same width")
    concatenated_image = np.vstack(images)
    return concatenated_image

def concatenate_images_horizontally(images):
    if len(set(image.shape[0] for image in images)) != 1:
        raise ValueError("Images must have the same height")
    concatenated_image = np.hstack(images)
    return concatenated_image

def concatenate_images_vertically_1(images):
    # Ensure all images have the same width
    max_width = max(image.shape[1] for image in images)
    resized_images = [cv2.resize(image, (max_width, image.shape[0])) for image in images]
    concatenated_image = np.vstack(resized_images)
    return concatenated_image

def concatenate_images_horizontally_1(images):
    # Ensure all images have the same height
    max_height = max(image.shape[0] for image in images)
    resized_images = [cv2.resize(image, (image.shape[1], max_height)) for image in images]
    concatenated_image = np.hstack(resized_images)
    return concatenated_image

images_with_predictions = []



# Loop through the test filenames and predictions
for filename, is_two_digits, digit_0, digit_1 in zip(test_filenames, is_two_digits_pred, digit_0_pred, digit_1_pred):
    print(f"Filename: {filename}")
    print(f"Is two digits: {is_two_digits}")
    print(f"Digit 0 prediction: {np.argmax(digit_0)}")
    print(
        f"Digit 1 prediction: {np.argmax(digit_1) if is_two_digits > 0.5 else 'None'}")  # Only print if two digits predicted

    # Read the image
    full_path = os.path.join(test_path, filename)
    image = cv2.imread(full_path)

    # Write predictions on the image
    image_with_predictions = write_predictions_on_image(image, is_two_digits, digit_0, digit_1)
    images_with_predictions.append(image_with_predictions)

    # # Display the image
    # cv2.imshow('Prediction', image_with_predictions)
    # cv2.waitKey(0)  # Wait for a key press to close the window

# Display grids with 9 images each
images_per_grid = 6
num_grids = (len(images_with_predictions) + images_per_grid - 1) // images_per_grid  # Calculate number of grids needed

for grid_index in range(num_grids):
    start_index = grid_index * images_per_grid
    end_index = min((grid_index + 1) * images_per_grid, len(images_with_predictions))
    grid_images = images_with_predictions[start_index:end_index]

    # Create a grid of images
    images_per_row = 3  # Number of images per row in the grid
    rows = [concatenate_images_horizontally(grid_images[i:i + images_per_row]) for i in range(0, len(grid_images), images_per_row)]
    grid_image = concatenate_images_vertically(rows)

    # Display the grid image
    cv2.imshow(f'Grid {grid_index + 1}', grid_image)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Destroy all OpenCV windows
cv2.destroyAllWindows()


