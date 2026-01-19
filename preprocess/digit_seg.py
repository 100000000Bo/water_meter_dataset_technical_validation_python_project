import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

# Path to the CSV file containing image names and processing flags
csv_file = "C:/Users/yibo/Desktop/detection/train/train_class_label_CSV.csv"

# Paths to original images and corresponding binary masks
original_folder = "C:/Users/yibo/Desktop/detection/train/train_img"
binary_folder = "C:/Users/yibo/Desktop/detection/train/train_seg_label"
output_folder = "output_images"

# Create output directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the CSV file
df = pd.read_csv(csv_file)

# Target output size
output_width, output_height = 200, 60

# Iterate through each row in the CSV file
for index, row in df.iterrows():
    image_name = row.iloc[0]
    process_flag = row.iloc[7]

    if process_flag == 1:
        original_image_path = os.path.join(original_folder, image_name)
        binary_image_path = os.path.join(binary_folder, image_name)

        original = cv2.imread(original_image_path)
        binary = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

        # Detect external contours from the binary mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Compute the minimum-area bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Determine orientation and define target rectangle corners
            a = rect[1][0]
            b = rect[1][1]

            if a < b:
                target_pts = np.array(
                    [[0, 0],
                     [output_width - 1, 0],
                     [output_width - 1, output_height - 1],
                     [0, output_height - 1]],
                    dtype="float32"
                )
            else:
                target_pts = np.array(
                    [[output_width - 1, 0],
                     [output_width - 1, output_height - 1],
                     [0, output_height - 1],
                     [0, 0]],
                    dtype="float32"
                )

            # Compute the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(box.astype("float32"), target_pts)

            # Apply perspective transformation to obtain a normalized image patch
            warped_image = cv2.warpPerspective(
                original, matrix, (output_width, output_height)
            )

            output_path = os.path.join(output_folder, f"segmented_{image_name}")
            cv2.imwrite(output_path, warped_image)

        print(f"Processed {image_name} and saved the result.")
