import json
import cv2
import os

json_path = 'C:/Users/HassenBELHASSEN/Desktop/test/jerseys/alphapose-results.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print(len(data[5]['keypoints']))
def extract_sub_images(json, image_folder , results):
    for item in json:
        image_id = item['image_id']
        keypoints = item['keypoints']

        # Open the image
        image_path = os.path.join(image_folder, image_id)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        rs_x, rs_y, rs_z = int(keypoints[3]), int(keypoints[4]), int(keypoints[5])
        ls_x, ls_y, ls_z = int(keypoints[12]), int(keypoints[13]), int(keypoints[14])
        rh_x, rh_y, rh_z = int(keypoints[21]), int(keypoints[22]), int(keypoints[23])
        lh_x, lh_y, lh_z = int(keypoints[30]), int(keypoints[31]), int(keypoints[32])

        points = [(rs_x, rs_y), (ls_x, ls_y), (rh_x, rh_y), (lh_x, lh_y)]

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # rs, ls, rh, lh = keypoints[2] , keypoints[5] , keypoints[8] , keypoints[11]
        # rs, ls, rh, lh = int(rs), int(ls), int(rh), int(lh)

        # Crop the image to get the sub-image
        sub_image = image[min_y:max_y, min_x:max_x]

        # Save the sub-image
        sub_image_path = os.path.join(results, f"sub_{image_id}")
        cv2.imwrite(sub_image_path, sub_image)
        print(f"Saved sub-image: {sub_image_path}")


# Assuming images are in a folder named 'images' on your desktop
image_folder = 'C:/Users/HassenBELHASSEN/Desktop/jerseys/jerseys'
outputs = 'C:/Users/HassenBELHASSEN/Desktop/testtest'

extract_sub_images(data, image_folder,outputs)