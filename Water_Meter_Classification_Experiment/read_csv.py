import csv
import random


def class2encode(which_class):
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
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == image_name:
                return int(row[class2encode(which_class)])
    return None  # 如果未找到对应的图片名，返回None


def select_images_for_training(csv_file, which_class, num_1, num_0):
    image_1 = []
    image_0 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]
            # print(class2encode(which_class))
            clarity_label = int(row[class2encode(which_class)])  # 假设第二列是清晰度标签
            if clarity_label == 1:
                image_1.append(image_name)
            elif clarity_label == 0:
                image_0.append(image_name)

    # 随机挑选指定数量的图片
    selected_1 = random.sample(image_1, min(num_1, len(image_1)))
    selected_0 = random.sample(image_0, min(num_0, len(image_0)))

    return selected_1, selected_0


def select_images_for_seg_testing(csv_file, which_class, num):
    image_1 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(image_1) >= num:
                break
            image_name = row[0]
            # print(class2encode(which_class))
            clarity_label = int(row[class2encode(which_class)])  # 假设第二列是清晰度标签
            if clarity_label == 1:
                image_1.append(image_name)

    return image_1

def select_images_for_dial_stained_training(csv_file, which_class, num_1, num_0):
    image_1 = []
    image_0 = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]
            # print(class2encode(which_class))
            clarity_label = int(row[class2encode(which_class)])  # 假设第二列是清晰度标签
            if clarity_label == 1:
                image_1.append(image_name)
            elif clarity_label == 0:
                image_0.append(image_name)

    # 随机挑选指定数量的图片
    selected_1 = random.sample(image_1, min(num_1, len(image_1)))
    selected_0 = random.sample(image_0, min(num_0, len(image_0)))

    return selected_1, selected_0
