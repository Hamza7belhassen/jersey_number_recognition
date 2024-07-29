# import easyocr
# import cv2
#
# reader = easyocr.Reader(['en'])
#
# path = 'C:/Users/HassenBELHASSEN/Desktop/alphapose_examples/1/alphapose_test_output/sub_citydort_2.jpg'
# img = cv2.imread(path)
# img = cv2.resize(img,(128,128))
# # img = cv2.medianBlur(img, 3)
# cv2.imshow('image' , img)
# cv2.waitKey(0)
#
# res = reader.readtext(path,allowlist='0123456789')
#
# print(res)

import os
import random
import cv2
import numpy as np
import uuid


# Function to generate a single digit image using cv2
def generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size):
    # Create a blank image with the specified background color
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * background_color

    apply_augmentations_cv2(image)
    # Choose a font and set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(digit)

    # Calculate text size and position to center the text
    text_size, _ = cv2.getTextSize(text, font, font_size, 1)
    text_x = (image_size[0] - text_size[0]) // 2
    text_y = (image_size[1] + text_size[1]) // 2

    # Draw the text onto the image
    cv2.putText(image, text, (text_x, text_y), font, font_size, font_color, 15)

    return image


# Function to apply augmentations using cv2
def apply_augmentations_cv2(image):
    # Example augmentation: Gaussian Noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise ,dtype=cv2.CV_8U)
    return noisy_image


# Generate Simple2D dataset using cv2
def generate_simple2d_dataset_cv2(output_dir, num_images_per_digit=500, image_size=(100, 100)):
    digits = list(range(100))
    background_colors = [(255, 0, 0), (0, 0, 128), (0, 128, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
    font_colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (0, 255, 0), (128, 0, 128)]
    font_sizes = [1,2,3,4]  # Add more font sizes as needed

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for digit in digits:
        for _ in range(num_images_per_digit):
            # Randomly select background color and font color
            background_color = random.choice(background_colors)
            font_color = random.choice(font_colors)

            while background_color == font_color:
                background_color = random.choice(background_colors)
                font_color = random.choice(font_colors)

            # Randomly select font size
            font_size = random.choice(font_sizes)

            # Generate the digit image using cv2
            digit_image = generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size)

            # Apply augmentations using cv2
            # augmented_image = apply_augmentations_cv2(digit_image)

            # Save the image
            image_filename = os.path.join(output_dir, f'{digit:02d}_{uuid.uuid4()}.png')
            cv2.imwrite(image_filename, digit_image)

            print(f'Saved {image_filename}')


# Example usage
output_directory = 'C:/Users/HassenBELHASSEN/Desktop/Simple2D_dataset'
# digit_image=generate_digit_image_cv2(7,(255,0,0),(0,0,0),10,(560,560))
# digit_image = digit_image.astype(np.uint8)
# cv2.imwrite('digit_image.png', digit_image)
# cv2.imshow('Digit Image', digit_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
generate_simple2d_dataset_cv2(output_directory)

# import os
# import random
# import cv2
# import numpy as np
#
#
# # Function to load random COCO image using cv2
# def load_random_coco_image_cv2(coco_images_dir):
#     coco_images = os.listdir(coco_images_dir)
#     random_image_path = os.path.join(coco_images_dir, random.choice(coco_images))
#     coco_image = cv2.imread(random_image_path)
#     return coco_image
#
#
# # Function to superimpose digit image onto COCO image using cv2
# def superimpose_digit_on_coco_cv2(coco_image, digit_image):
#     # Resize digit image to fit a random position on COCO image
#     digit_resized = cv2.resize(digit_image, (100, 100))  # Resize to match your requirements
#
#     # Choose a random position to place the digit image on the COCO image
#     pos_x = random.randint(0, coco_image.shape[1] - digit_resized.shape[1])
#     pos_y = random.randint(0, coco_image.shape[0] - digit_resized.shape[0])
#
#     # Create a mask to blend the digit image onto the COCO image
#     mask = (digit_resized != 0)
#
#     # Blend the images using the mask
#     blended_image = coco_image.copy()
#     blended_image[pos_y:pos_y + digit_resized.shape[0], pos_x:pos_x + digit_resized.shape[1]][mask] = digit_resized[
#         mask]
#
#     return blended_image
#
#
# # Generate Complex2D dataset using cv2
# def generate_complex2d_dataset_cv2(output_dir, coco_images_dir, num_images_per_digit=4000):
#     digits = list(range(100))
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     for digit in digits:
#         for _ in range(num_images_per_digit):
#             # Load a random COCO image
#             coco_image = load_random_coco_image_cv2(coco_images_dir)
#
#             # Generate and superimpose the digit image
#             digit_image = generate_digit_image_cv2(digit, (255, 255, 255), (0, 0, 0), 60,
#                                                    (100, 100))  # Example parameters
#             complex_image = superimpose_digit_on_coco_cv2(coco_image, digit_image)
#
#             # Save the image
#             image_filename = os.path.join(output_dir, f'{digit:02d}_{uuid.uuid4()}.png')
#             cv2.imwrite(image_filename, complex_image)
#
#             print(f'Saved {image_filename}')
#
#
# # Example usage
# output_directory = 'path_to_output_directory_for_complex2d_dataset'
# coco_images_directory = 'path_to_coco_images_directory'
# generate_complex2d_dataset_cv2(output_directory, coco_images_directory)
