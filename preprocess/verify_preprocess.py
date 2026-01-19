# Remove the existing result directory before regeneration
# Example command:
# python labelme2voc.py label_json result --labels labels.txt --noobject

import os
import shutil
import cv2

label_from_PATH = "result/SegmentationClass"
label_to_PATH = "result/binary_label"

filepath_list = os.listdir(label_from_PATH)
# filepath_list.remove("labelme_json_to_dataset.exe")

# Recreate the binary label directory
if os.path.isdir(label_to_PATH):
    shutil.rmtree(label_to_PATH)
os.mkdir(label_to_PATH)

bin_img = True

for i, file_path in enumerate(filepath_list):
    src_label = "{}".format(os.path.join(label_from_PATH, filepath_list[i]))
    label_name = "{}".format(file_path)

    if bin_img:
        dest_label = cv2.imread(src_label)
        dest_label = cv2.cvtColor(dest_label, cv2.COLOR_BGR2GRAY)
        ret, dest_label = cv2.threshold(dest_label, 0, 255, cv2.THRESH_BINARY)
        cv2.imwrite(
            os.path.join(label_to_PATH, label_name),
            dest_label,
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )
    else:
        shutil.copy(src_label, os.path.join(label_to_PATH, label_name))

    print("{} has been copied to {}".format(label_name, label_to_PATH))

print("All done!")


# Path to the directory containing original JPEG images
jpg_path = 'result/JPEGImages'

# Convert image file extensions from .jpg to .png
for filename in os.listdir(jpg_path):
    if filename.endswith('.jpg'):
        new_filename = filename[:-4] + '.png'
        old_file = os.path.join(jpg_path, filename)
        new_file = os.path.join(jpg_path, new_filename)
        os.rename(old_file, new_file)

print("File extensions have been successfully updated.")
