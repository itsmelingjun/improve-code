import numpy as np
from PIL import ImageFont
from PIL import Image
from stardist.models import StarDist2D

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

# 加载预训练的 StarDist2D 模型
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 获取每个像素点的标签值
def get_margin(input):
    if isinstance(input, Image.Image):
        img = np.array(input)
    else:
        img = imread(input)
    # 将图像转换为灰度并归一化
    if img.ndim == 3:  # 如果是 RGB 图像，则转换为灰度图
        img = rgb2gray(img)
    img = normalize(img)  # 将图像归一化
    # 使用模型进行实例分割预测
    labels, _ = model.predict_instances(img)
    return labels

# 检查CenterNet检测框中超过80%的区域被包围的物体，调整检测框以确保物体完整包围
def bounding_box_correction(detection_boxes, labels, img_shape,threshold1=0.1 , threshold2=0.7 , threshold3=0.05):
    correct_boxes = []
    # 获取图像的宽度和高度
    img_width, img_height = img_shape
    print(img_width, img_height)
    # 遍历每个检测框
    for box in detection_boxes:
        y_min, x_min, y_max, x_max = box
        print(box)
        # 强制转换为整数
        x_min = int(np.floor(x_min))
        y_min = int(np.floor(y_min))
        x_max = int(np.floor(x_max))  # 保证原始框全部被包含在内
        y_max = int(np.floor(y_max))
        # 确保索引不越界
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = min(x_max, img_width)
        y_max = min(y_max, img_height)
        print(y_min, x_min, y_max, x_max)

        detection_area = (x_max - x_min) * (y_max - y_min)

        # 统计检测框内的标签值
        box_labels = labels[y_min:y_max, x_min:x_max]
        # 获取检测框内包含的每个实例标签
        unique_labels = np.unique(box_labels)

        for label in unique_labels:
            if label == 0:  # 跳过背景标签
                continue
            part_area = np.sum(box_labels == label)  # 计算每个标签代表物体在框内的面积
            total_area = np.sum(labels == label)  # 计算每个标签代表物体在原图内的面积
            # 计算物体在检测框内的覆盖比例
            area_ratio = part_area / detection_area
            coverage_ratio = part_area / total_area
            # 如果覆盖度超过阈值，调整检测框以确保物体完整包围
            if area_ratio >= threshold1 and coverage_ratio >= threshold2:
                # 获取物体的边界
                object_margin = np.isin(labels, label)  # 判断原图中哪些区域是指定元素
                coords = np.column_stack(np.where(object_margin))  # 获取指定元素所在位置索引
                # 找到物体区域的角点
                y_min_new, x_min_new = coords.min(axis=0)
                y_max_new, x_max_new = coords.max(axis=0)
                # 将新的边界框放大，确保完整包围物体并且新的边界框在图像边界内
                x_min = max(0, min(x_min, x_min_new))
                y_min = max(0, min(y_min, y_min_new))
                x_max = min(img_shape[0], max(x_max, x_max_new))
                y_max = min(img_shape[1], max(y_max, y_max_new))

            # 如果物体被边框穿过并且占比小于阈值，则调整边界框
            if area_ratio < threshold3 and 0 < part_area < total_area:
                object_margin = (labels == label)  # 获取整个物体的范围
                coords = np.column_stack(np.where(object_margin))  # 获取物体的像素坐标

                # 获取物体上下左右的边界
                y_min_obj, x_min_obj = coords.min(axis=0)
                y_max_obj, x_max_obj = coords.max(axis=0)

                # 收缩边界框的边界
                y_min = max(y_min_obj, y_min)
                x_min = max(x_min_obj, x_min)
                y_max = min(y_max_obj, y_max)
                x_max = min(x_max_obj, x_max)

        correct_boxes.append((y_min, x_min, y_max, x_max))

    print(correct_boxes)
    return correct_boxes

