import csv
import random


def class2encode(which_class):
    """Map class name to corresponding column index in CSV."""
    if which_class == 'clear':
        return 1
    if which_class == 'blurry':
        return 2
    if which_class == 'dial-stained':
        return 3
    if which_class == 'soil-covered':
        return 4
    if which_class == 'dark':
        return 5
    if which_class == 'reflective':
        return 6
    if which_class == 'six-digit':
        return 7


def get_value_from_csv(csv_file, image_name, which_class):
    """Get the label value for a specific image and class from CSV."""
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == image_name:
                return int(row[class2encode(which_class))])
    return None  # Return None if image name is not found


def select_images_for_training(csv_file, which_class, num_1, num_0):
    """Select a specified number of positive (1) and negative (0) samples for training."""
    image_1 = []
    image_0 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]
            label = int(row[class2encode(which_class)])
            if label == 1:
                image_1.append(image_name)
            elif label == 0:
                image_0.append(image_name)

    selected_1 = random.sample(image_1, min(num_1, len(image_1)))
    selected_0 = random.sample(image_0, min(num_0, len(image_0)))

    return selected_1, selected_0


def select_images_for_seg_testing(csv_file, which_class, num):
    """Select a specified number of positive samples for segmentation testing."""
    image_1 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(image_1) >= num:
                break
            image_name = row[0]
            label = int(row[class2encode(which_class)])
            if label == 1:
                image_1.append(image_name)

    return image_1


def select_images_for_dial_stained_training(csv_file, which_class, num_1, num_0):
    """Select a specified number of positive and negative samples for dial-stained training."""
    image_1 = []
    image_0 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]
            label = int(row[class2encode(which_class)])
            if label == 1:
                image_1.append(image_name)
            elif label == 0:
                image_0.append(image_name)

    selected_1 = random.sample(image_1, min(num_1, len(image_1)))
    selected_0 = random.sample(image_0, min(num_0, len(image_0)))

    return selected_1, selected_0
