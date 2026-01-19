import os
from PIL import Image


def resize_images_in_folder(input_folder, output_folder, new_width, new_height):
    # Iterate through all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct input and output image paths
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            # Resize the image to the specified resolution
            resize_image(input_image_path, output_image_path, new_width, new_height)


def resize_image(input_image_path, output_image_path, new_width, new_height):
    image = Image.open(input_image_path)
    resized_image = image.resize((new_width, new_height))
    resized_image.save(output_image_path)


input_folder = "clear_all"
output_folder = "clear_512size"
new_width = 512
new_height = 512

resize_images_in_folder(input_folder, output_folder, new_width, new_height)

input_folder = "challenge_all"
output_folder = "challenge_512size"
new_width = 512
new_height = 512

resize_images_in_folder(input_folder, output_folder, new_width, new_height)
