import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import RoIAlign

class Build(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone: 预训练ResNet50去掉最后两层（avgpool和fc）
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 输出C5特征 (2048通道)
        
        # RoIAlign层，输出大小固定为7x7
        self.roi_align = RoIAlign(output_size=(7,7), spatial_scale=1/32, sampling_ratio=2)  
        # spatial_scale = 输入图像和特征图尺寸比，ResNet50下C5大约缩小32倍
        
        # 分类和边框回归头，全连接层
        self.fc1 = nn.Linear(2048*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.cls_score = nn.Linear(1024, num_classes)    # 分类
        self.bbox_pred = nn.Linear(1024, num_classes*4) # 边框回归
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, images, rois):
        """
        images: Tensor (N, 3, H, W)
        rois: Tensor (num_rois, 5) [batch_index, x1, y1, x2, y2]
        """
        # 提取特征
        features = self.backbone(images)  # (N, 2048, H/32, W/32)
        
        # RoIAlign操作，从特征图中裁剪对应RoI区域 (num_rois, 2048, 7, 7)
        roi_features = self.roi_align(features, rois)
        
        # 平展
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(roi_features))
        x = self.relu(self.fc2(x))
        
        # 输出类别分数和边框回归
        cls_logits = self.cls_score(x)           # (num_rois, num_classes)
        bbox_preds = self.bbox_pred(x)           # (num_rois, num_classes*4)
        
        return cls_logits, bbox_preds
