# import os
# import random
# import uuid
# import cv2
# import numpy as np
#
#
# # Function to generate a single digit image using cv2
# def generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size):
#     # Create a blank image with the specified background color
#     image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
#
#     # Choose a font and set font properties
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text = str(digit)
#
#     # Calculate text size and position to center the text
#     text_size, _ = cv2.getTextSize(text, font, font_size, 2)
#     text_x = (image_size[0] - text_size[0]) // 2
#     text_y = (image_size[1] + text_size[1]) // 2
#
#     # Draw the text onto the image
#     cv2.putText(image, text, (text_x, text_y), font, font_size, font_color, 10)
#
#     return image
#
#
# # Function to apply light augmentations using cv2
# def apply_light_augmentations(image):
#     # Create a mask where the digit is located
#     digit_mask = cv2.inRange(image, np.array([1, 1, 1]), np.array([255, 255, 255]))
#
#     # Extract the digit region
#     digit_region = cv2.bitwise_and(image, image, mask=digit_mask)
#
#     # Convert to float for more complex augmentations
#     digit_region = digit_region.astype(np.float32)
#
#     # Gaussian Noise
#     noise = np.random.normal(0, 10, digit_region.shape).astype(np.float32)
#     noisy_image = cv2.add(digit_region, noise)
#
#     # Optical Distortion
#     height, width = noisy_image.shape[:2]
#     distortion = np.random.normal(0, 1, (height, width, 2)).astype(np.float32)
#     distorted_image = cv2.remap(noisy_image,
#                                 np.float32(np.indices((height, width)).transpose(1, 2, 0) + distortion),
#                                 None,
#                                 cv2.INTER_LINEAR)
#
#     # Convert back to uint8
#     distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
#
#     # Composite the augmented digit back onto the original image
#     background_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(digit_mask))
#     augmented_image = cv2.add(background_region, distorted_image)
#
#     return augmented_image
#
#
# # Function to apply medium augmentations using cv2
# def apply_medium_augmentations(image):
#     # Apply light augmentations first
#     image = apply_light_augmentations(image)
#
#     # Create a mask where the digit is located
#     digit_mask = cv2.inRange(image, np.array([1, 1, 1]), np.array([255, 255, 255]))
#
#     # Extract the digit region
#     digit_region = cv2.bitwise_and(image, image, mask=digit_mask)
#
#     # Convert to float for more complex augmentations
#     digit_region = digit_region.astype(np.float32)
#
#     # Grid Distortion
#     height, width = digit_region.shape[:2]
#     num_cells = 10
#     cell_size_x = width // num_cells
#     cell_size_y = height // num_cells
#
#     map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
#     map_x = map_x.astype(np.float32)
#     map_y = map_y.astype(np.float32)
#
#     distortions = np.random.uniform(-5, 5, (num_cells + 1, num_cells + 1, 2)).astype(np.float32)
#
#     for i in range(num_cells):
#         for j in range(num_cells):
#             start_x = i * cell_size_x
#             start_y = j * cell_size_y
#             end_x = start_x + cell_size_x
#             end_y = start_y + cell_size_y
#
#             if end_x > width:
#                 end_x = width
#             if end_y > height:
#                 end_y = height
#
#             dx1, dy1 = distortions[i, j]
#             dx2, dy2 = distortions[i + 1, j]
#             dx3, dy3 = distortions[i, j + 1]
#             dx4, dy4 = distortions[i + 1, j + 1]
#
#             map_x[start_y:end_y, start_x:end_x] = (
#                 np.linspace(start_x + dx1, end_x + dx2, end_x - start_x)[:, None] +
#                 np.linspace(0, dx3 - dx1, end_y - start_y)[None, :]
#             )
#             map_y[start_y:end_y, start_x:end_x] = (
#                 np.linspace(start_y + dy1, end_y + dy2, end_y - start_y)[None, :] +
#                 np.linspace(0, dy4 - dy1, end_x - start_x)[:, None]
#             )
#
#     distorted_digit_region = cv2.remap(digit_region, map_x, map_y, interpolation=cv2.INTER_LINEAR)
#
#     # Convert back to uint8
#     digit_region = np.clip(distorted_digit_region, 0, 255).astype(np.uint8)
#
#     # Composite the augmented digit back onto the original image
#     background_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(digit_mask))
#     augmented_image = cv2.add(background_region, digit_region)
#
#     return augmented_image
#
#
#
#
# # Function to apply hard augmentations using cv2
# def apply_hard_augmentations(image):
#     # Apply medium augmentations first
#     image = apply_medium_augmentations(image)
#
#     # Create a mask where the digit is located
#     digit_mask = cv2.inRange(image, np.array([1, 1, 1]), np.array([255, 255, 255]))
#
#     # Extract the digit region
#     digit_region = cv2.bitwise_and(image, image, mask=digit_mask)
#
#     # Convert to float for more complex augmentations
#     digit_region = digit_region.astype(np.float32)
#
#     # Shuffling RGB Channels
#     shuffled_image = digit_region.copy()
#     channels = list(cv2.split(shuffled_image))  # Convert to list
#     random.shuffle(channels)  # Shuffle the list
#     shuffled_image = cv2.merge(channels)  # Convert back to tuple
#
#     # Random Shift-Scale-Rotation
#     height, width = shuffled_image.shape[:2]
#     center = (width // 2, height // 2)
#     angle = random.uniform(-30, 30)
#     scale = random.uniform(0.7, 1.3)
#     shift_x = random.uniform(-10, 10)
#     shift_y = random.uniform(-10, 10)
#
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
#     rotation_matrix[0, 2] += shift_x
#     rotation_matrix[1, 2] += shift_y
#
#     augmented_digit_region = cv2.warpAffine(shuffled_image, rotation_matrix, (width, height))
#
#     # Convert back to uint8
#     augmented_digit_region = np.clip(augmented_digit_region, 0, 255).astype(np.uint8)
#
#     # Composite the augmented digit back onto the original image
#     background_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(digit_mask))
#     augmented_image = cv2.add(background_region, augmented_digit_region)
#
#     return augmented_image
#
#
# # Generate Simple2D dataset using cv2
# def generate_simple2d_dataset_cv2(output_dir, num_images_per_digit=4000, image_size=(100, 100)):
#     digits = list(range(100))
#     background_colors = [(255, 0, 0), (0, 0, 128), (0, 128, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
#     font_colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (0, 255, 0), (128, 0, 128)]
#     font_sizes = [1,2,3]  # Adjusted font sizes for cv2
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     for digit in digits:
#         # Generate 1 light augmented image
#         background_color = random.choice(background_colors)
#         font_color = random.choice(font_colors)
#         while background_color == font_color:
#             background_color = random.choice(background_colors)
#             font_color = random.choice(font_colors)
#         font_size = random.choice(font_sizes)
#         digit_image = generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size)
#         augmented_image = apply_light_augmentations(digit_image)
#         image_filename = os.path.join(output_dir, f'{digit:02d}_light_{uuid.uuid4()}.png')
#         cv2.imwrite(image_filename, augmented_image)
#         print(f'Saved {image_filename}')
#
#         # Generate 5 medium augmented images
#         for _ in range(5):
#             background_color = random.choice(background_colors)
#             font_color = random.choice(font_colors)
#             font_size = random.choice(font_sizes)
#             digit_image = generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size)
#             augmented_image = apply_medium_augmentations(digit_image)
#             image_filename = os.path.join(output_dir, f'{digit:02d}_medium_{uuid.uuid4()}.png')
#             cv2.imwrite(image_filename, augmented_image)
#             print(f'Saved {image_filename}')
#
#         # Generate 5 hard augmented images
#         for _ in range(5):
#             background_color = random.choice(background_colors)
#             font_color = random.choice(font_colors)
#             font_size = random.choice(font_sizes)
#             digit_image = generate_digit_image_cv2(digit, background_color, font_color, font_size, image_size)
#             augmented_image = apply_hard_augmentations(digit_image)
#             image_filename = os.path.join(output_dir, f'{digit:02d}_hard_{uuid.uuid4()}.png')
#             cv2.imwrite(image_filename, augmented_image)
#             print(f'Saved {image_filename}')


# import os
# import random
# import uuid
# import cv2
# import numpy as np
#
#
# # Function to generate a single digit image with a transparent background using cv2
# def generate_digit_image_cv2(digit, font_color, font_size, image_size):
#     # Create a blank image with a transparent background
#     image = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)
#
#     # Choose a font and set font properties
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text = str(digit)
#
#     # Calculate text size and position to center the text
#     text_size, _ = cv2.getTextSize(text, font, font_size, 2)
#     text_x = (image_size[0] - text_size[0]) // 2
#     text_y = (image_size[1] + text_size[1]) // 2
#
#     # Draw the text onto the image
#     cv2.putText(image, text, (text_x, text_y), font, font_size, font_color, 2, lineType=cv2.LINE_AA)
#
#     # Set alpha channel
#     alpha_channel = np.where(image[:, :, :3].any(axis=2), 255, 0).astype(np.uint8)
#     image[:, :, 3] = alpha_channel
#
#     return image
#
#
# # Function to apply augmentations using cv2
# def apply_augmentations_cv2(image):
#     # Convert to float for more complex augmentations
#     image = image.astype(np.float32)
#
#     # 1. Gaussian Noise
#     noise = np.random.normal(0, 10, image.shape).astype(np.float32)
#     noisy_image = cv2.add(image, noise)
#
#     # 2. Optical Distortion
#     height, width = noisy_image.shape[:2]
#     distortion = np.random.normal(0, 1, (height, width, 2)).astype(np.float32)
#     distorted_image = cv2.remap(noisy_image,
#                                 np.float32(np.indices((height, width)).transpose(1, 2, 0) + distortion),
#                                 None,
#                                 cv2.INTER_LINEAR)
#
#     # 3. Grid Distortion
#     num_cells = 10
#     cell_size_x = width // num_cells
#     cell_size_y = height // num_cells
#     distortions = np.random.uniform(-10, 10, (num_cells + 1, num_cells + 1, 2)).astype(np.float32)
#
#     for i in range(num_cells):
#         for j in range(num_cells):
#             start_x = i * cell_size_x
#             start_y = j * cell_size_y
#             end_x = start_x + cell_size_x
#             end_y = start_y + cell_size_y
#
#             pts1 = np.array([[start_x, start_y],
#                              [end_x, start_y],
#                              [start_x, end_y]], dtype=np.float32)
#
#             pts2 = pts1 + distortions[i:i + 2, j:j + 2].reshape(-1, 2)
#             matrix = cv2.getAffineTransform(pts1, pts2)
#             distorted_image[start_y:end_y, start_x:end_x] = cv2.warpAffine(
#                 distorted_image[start_y:end_y, start_x:end_x], matrix, (cell_size_x, cell_size_y))
#
#
#     # 4. Shuffling RGB Channels
#     shuffled_image = distorted_image.copy()
#     channels = cv2.split(shuffled_image)
#     random.shuffle(channels)
#     shuffled_image = cv2.merge(channels)
#
#     # 5. Random Shift-Scale-Rotation
#     center = (width // 2, height // 2)
#     angle = random.uniform(-30, 30)
#     scale = random.uniform(0.7, 1.3)
#     shift_x = random.uniform(-10, 10)
#     shift_y = random.uniform(-10, 10)
#
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
#     rotation_matrix[0, 2] += shift_x
#     rotation_matrix[1, 2] += shift_y
#
#     augmented_image = cv2.warpAffine(shuffled_image, rotation_matrix, (width, height))
#
#     # Convert back to uint8
#     augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
#
#     return augmented_image
#
#
# # Function to composite digit image onto background
# def composite_digit_on_background(digit_image, background_color, image_size):
#     # Create a blank background image with the specified color
#     background = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
#
#     # Split the digit image into BGR and alpha channels
#     bgr_image = digit_image[:, :, :3]
#     alpha_channel = digit_image[:, :, 3]
#
#     # Composite the digit image onto the background
#     alpha_factor = alpha_channel[:, :, None] / 255.0
#     composite_image = background * (1 - alpha_factor) + bgr_image * alpha_factor
#     composite_image = composite_image.astype(np.uint8)
#
#     return composite_image
#
#
# # Generate Simple2D dataset using cv2
# def generate_simple2d_dataset_cv2(output_dir, num_images_per_digit=4000, image_size=(100, 100)):
#     digits = list(range(100))
#     background_colors = [(255, 0, 0), (0, 0, 128), (0, 128, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
#     font_colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (0, 255, 0), (128, 0, 128)]
#     font_sizes = [0.5, 1, 1.5]  # Adjusted font sizes for cv2
#
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     for digit in digits:
#         for _ in range(num_images_per_digit):
#             # Randomly select background color and font color
#             background_color = random.choice(background_colors)
#             font_color = random.choice(font_colors)
#
#             # Randomly select font size
#             font_size = random.choice(font_sizes)
#
#             # Generate the digit image using cv2
#             digit_image = generate_digit_image_cv2(digit, font_color, font_size, image_size)
#
#             # Apply augmentations to the digit image
#             augmented_digit_image = apply_augmentations_cv2(digit_image)
#
#             # Composite the augmented digit onto the background
#             composite_image = composite_digit_on_background(augmented_digit_image, background_color, image_size)
#
#             # Save the image
#             image_filename = os.path.join(output_dir, f'{digit:02d}_{uuid.uuid4()}.png')
#             cv2.imwrite(image_filename, composite_image)
#
#             print(f'Saved {image_filename}')



import os
import random
import uuid
import cv2
import numpy as np


# Function to generate a single digit image with alpha channel using cv2
def generate_digit_image_cv2(digit, font_color, font_size, image_size):
    # Create a blank image with a transparent background
    image = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)

    # Choose a font and set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(digit)

    # Calculate text size and position to center the text
    text_size, _ = cv2.getTextSize(text, font, font_size, 2)
    text_x = (image_size[0] - text_size[0]) // 2
    text_y = (image_size[1] + text_size[1]) // 2

    # Draw the text onto the image
    cv2.putText(image, text, (text_x, text_y), font, font_size, font_color, 10)

    # Set the alpha channel where the text is
    alpha_channel = np.where((image[:, :, 0] > 0) | (image[:, :, 1] > 0) | (image[:, :, 2] > 0), 255, 0).astype(np.uint8)
    image[:, :, 3] = alpha_channel

    return image


# Function to apply light augmentations using cv2
def apply_light_augmentations(digit_image):
    # Gaussian Noise
    noise = np.random.normal(0, 5, digit_image[:, :, :3].shape).astype(np.float32)
    noisy_image = cv2.add(digit_image[:, :, :3].astype(np.float32), noise)

    # height, width = noisy_image.shape[:2]
    # distortion = np.random.normal(0, 0.5, (height, width, 2)).astype(np.float32)  # Reduced distortion
    # distorted_image = cv2.remap(noisy_image, np.float32(np.indices((height, width)).transpose(1, 2, 0) + distortion), None, cv2.INTER_LINEAR)
    # Combine with alpha channel
    noisy_image = np.dstack((noisy_image, digit_image[:, :, 3]))

    return noisy_image.astype(np.uint8)


# Function to apply medium augmentations using cv2
def apply_medium_augmentations(digit_image):
    # Apply light augmentations first
    image = apply_light_augmentations(digit_image)

    # Grid Distortion
    height, width = image.shape[:2]
    num_cells = 5
    cell_size_x = width // num_cells
    cell_size_y = height // num_cells

    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    distortions = np.random.uniform(-15, 15, (num_cells + 1, num_cells + 1, 2)).astype(np.float32)

    for i in range(num_cells):
        for j in range(num_cells):
            start_x = i * cell_size_x
            start_y = j * cell_size_y
            end_x = start_x + cell_size_x
            end_y = start_y + cell_size_y

            if end_x > width:
                end_x = width
            if end_y > height:
                end_y = height

            dx1, dy1 = distortions[i, j]
            dx2, dy2 = distortions[i + 1, j]
            dx3, dy3 = distortions[i, j + 1]
            dx4, dy4 = distortions[i + 1, j + 1]

            map_x[start_y:end_y, start_x:end_x] = (
                np.linspace(start_x + dx1, end_x + dx2, end_x - start_x)[:, None] +
                np.linspace(0, dx3 - dx1, end_y - start_y)[None, :]
            )
            map_y[start_y:end_y, start_x:end_x] = (
                np.linspace(start_y + dy1, end_y + dy2, end_y - start_y)[None, :] +
                np.linspace(0, dy4 - dy1, end_x - start_x)[:, None]
            )

    distorted_digit_region = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return distorted_digit_region.astype(np.uint8)


# Function to apply hard augmentations using cv2
def apply_hard_augmentations(digit_image):
    # Apply medium augmentations first
    image = apply_light_augmentations(digit_image)

    # Shuffling RGB Channels
    shuffled_image = image.copy()
    channels = list(cv2.split(shuffled_image[:, :, :3]))  # Exclude alpha channel
    random.shuffle(channels)
    shuffled_image[:, :, :3] = cv2.merge(channels)

    # Random Shift-Scale-Rotation
    height, width = shuffled_image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.9, 1.1)
    shift_x = random.uniform(-5, 5)
    shift_y = random.uniform(-5, 5)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_matrix[0, 2] += shift_x
    rotation_matrix[1, 2] += shift_y

    augmented_digit_region = cv2.warpAffine(shuffled_image[:, :, :3], rotation_matrix, (width, height))

    # Combine with alpha channel
    augmented_digit_region = np.dstack((augmented_digit_region, shuffled_image[:, :, 3]))

    return augmented_digit_region.astype(np.uint8)


# Function to compose final image with background
def compose_final_image(augmented_digit, background_color):
    # Create a blank background image
    background_color = list(background_color)
    background_image = np.ones_like(augmented_digit) * np.array(background_color + [255], dtype=np.uint8)

    # Blend the augmented digit onto the background using alpha blending
    alpha = augmented_digit[:, :, 3] / 255.0
    for c in range(3):
        background_image[:, :, c] = alpha * augmented_digit[:, :, c] + (1 - alpha) * background_image[:, :, c]

    # Set the alpha channel for the composed image
    background_image[:, :, 3] = 255

    return background_image.astype(np.uint8)


# Generate Simple2D dataset using cv2
def generate_simple2d_dataset_cv2(output_dir, num_images_per_digit=4000, image_size=(100, 100)):
    digits = list(range(100))
    background_colors = [ (255, 0, 0),(255, 255, 255), (0, 0, 0), (128, 0, 128), (255, 255, 0),(0, 0, 255), (0, 255, 0)]
    font_colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0), (0, 255, 0), (128, 0, 128)]
    font_sizes = [1, 2, 2.5]  # Adjusted font sizes for cv2

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for digit in digits:
        for _ in range(2):
            background_color = random.choice(background_colors)
            font_color = random.choice(font_colors)
            while True:
                background_color = random.choice(background_colors)
                font_color = random.choice(font_colors)
                if font_color != background_color:
                    break
            font_size = random.choice(font_sizes)
            digit_image = generate_digit_image_cv2(digit, font_color, font_size, image_size)
            composed_image = compose_final_image(digit_image, background_color)
            image_filename = os.path.join(output_dir, f'{digit:02d}_light_{uuid.uuid4()}.png')
            cv2.imwrite(image_filename, composed_image)
            print(f'Saved {image_filename}')

        # Generate 1 light augmented image
        background_color = random.choice(background_colors)
        font_color = random.choice(font_colors)
        font_size = random.choice(font_sizes)
        while True:
            background_color = random.choice(background_colors)
            font_color = random.choice(font_colors)
            if font_color != background_color:
                break
        digit_image = generate_digit_image_cv2(digit, font_color, font_size, image_size)
        augmented_image = apply_light_augmentations(digit_image)
        composed_image = compose_final_image(augmented_image, background_color)
        image_filename = os.path.join(output_dir, f'{digit:02d}_light_{uuid.uuid4()}.png')
        cv2.imwrite(image_filename, composed_image)
        print(f'Saved {image_filename}')

        # #Generate 5 medium augmented images
        for _ in range(5):
            background_color = random.choice(background_colors)
            font_color = random.choice(font_colors)
            font_size = random.choice(font_sizes)
            digit_image = generate_digit_image_cv2(digit, font_color, font_size, image_size)
            augmented_image = apply_medium_augmentations(digit_image)
            composed_image = compose_final_image(augmented_image, background_color)
            image_filename = os.path.join(output_dir, f'{digit:02d}_medium_{uuid.uuid4()}.png')
            cv2.imwrite(image_filename, composed_image)
            print(f'Saved {image_filename}')

        #Generate 5 hard augmented images
        for _ in range(5):
            background_color = random.choice(background_colors)
            font_color = random.choice(font_colors)
            font_size = random.choice(font_sizes)
            while background_color == font_color:
                background_color = random.choice(background_colors)
                font_color = random.choice(font_colors)
            digit_image = generate_digit_image_cv2(digit, font_color, font_size, image_size)
            augmented_image = apply_hard_augmentations(digit_image)
            composed_image = compose_final_image(augmented_image, background_color)
            image_filename = os.path.join(output_dir, f'{digit:02d}_hard_{uuid.uuid4()}.png')
            cv2.imwrite(image_filename, composed_image)
            print(f'Saved {image_filename}')


# Example usage
output_directory = 'C:/Users/HassenBELHASSEN/Desktop/Simple2D_dataset'
generate_simple2d_dataset_cv2(output_directory)

