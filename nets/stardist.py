import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from stardist.models import StarDist2D

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 加载预训练的StarDist2D模型
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 使用StarDist模型进行实例分割获取区域建议框并映射到特征图上
class stardist_Proposal(nn.Module):
    def __init__(self, feature_map_size=(64, 64), model=None):
        super(stardist_Proposal, self).__init__()
        self.feature_map_size = feature_map_size
        self.model = model  # StarDist 模型实例

    def stardist(self, image):
        # 转换为 RGB 格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将图像传入 StarDist 模型进行预测
        labels, _ = self.model.predict_instances(image_rgb)
        return labels

    def get_bounding_boxes(labels):
        """
        从StarDist模型输出的实例分割结果中提取边界框参数并筛选
        """
        # 获取图像的宽度和高度
        img_width, img_height = labels.size
        img_area = img_width * img_height
        bboxes = []
        # 遍历所有标签
        for label in np.unique(labels):

            if label == 0:  # 跳过背景
                continue

            # 获取每个物体的坐标
            coords = np.column_stack(np.where(labels == label))
            # 获取边界框参数（y_min, x_min, y_max, x_max）
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            obj_area = (x_max - x_min) * (y_max - y_min)
            ratio = obj_area / img_area

            if ratio < 0.00023:  # 跳过极小卵石
                continue

            # 存储边界框参数
            bboxes.append([y_min, x_min, y_max, x_max])

        return np.array(bboxes) , img_area

    def generate_proposals(bboxes, img_area, iou_threshold=0.5):
        """
        基于提取到的边界框参数排列组合生成生成预选框
        bboxes: 从实例分割提取的物体边界框参数
        """
        Top_lefts = []
        Bottom_rights = []
        # 将每个边界框的左上角和右下角分别存入两个列表
        for bbox in bboxes:
            y_min, x_min, y_max, x_max = bbox
            Top_lefts.append((y_min, x_min))
            Bottom_rights.append((y_max, x_max))

        proposals = []
        # 组合生成边界框
        for i in range(0, len(Top_lefts)):
            for j in range(i+1, len(Top_lefts)):
                proposals_area = (Bottom_rights[j][1] - Top_lefts[i][1]) * (Bottom_rights[j][0] - Top_lefts[i][0])
                threshold = 0.01555
                # 面积占比初筛
                if proposals_area/img_area >= threshold:
                    proposals.append((Top_lefts[i][0], Top_lefts[i][1], Bottom_rights[j][0], Bottom_rights[j][1]))

        return proposals

    def get_proposals_fp(proposals, upsampled_features, image):
        img_height, img_width = image.size
        feature_map_height, feature_map_width = upsampled_features.size

        # 计算缩放因子
        scale_x = feature_map_width / img_width
        scale_y = feature_map_height / img_height

        # 将边界框坐标映射到特征图的坐标系中
        mapped_bboxes = []
        for bbox in proposals:
            y_min, x_min, y_max, x_max = bbox
            # 将坐标缩放到特征图坐标系
            y_min_fm = int(y_min * scale_y)
            x_min_fm = int(x_min * scale_x)
            y_max_fm = int(y_max * scale_y)
            x_max_fm = int(x_max * scale_x)

            mapped_bboxes.append([y_min_fm, x_min_fm, y_max_fm, x_max_fm])

        return mapped_bboxes

    def forward(self, x, image):
        labels = self.stardist(image)
        bboxes, img_area = self.get_bounding_boxes(labels)
        proposals = self.generate_proposals(bboxes, img_area)
        mapped_bboxes = self.get_proposals_fp(proposals , x ,image)

        return mapped_bboxes

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        # 使用上采样恢复到原图大小
        upsampled_features = self.upsample(x3)
        return upsampled_features

class stardist_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(stardist_Head, self).__init__()
