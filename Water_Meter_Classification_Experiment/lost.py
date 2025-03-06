import os

# 设定图片所在的文件夹路径
folder_path = "C:/Users/yibo/Desktop/detection/train/train_img"

# 获取文件夹中所有的图片文件名
files = os.listdir(folder_path)

# 提取文件名中的数字部分
numbers = sorted([int(f[5:-4]) for f in files if f.startswith("train") and f.endswith(".png")])

# 找出缺失的序号
missing_numbers = [i for i in range(32713) if i not in numbers]

# 打印略过的序号
print("Missing numbers:", missing_numbers)