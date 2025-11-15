import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class RetinaNetHead(nn.Module):
    def __init__(self, input_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.cls_logits = nn.Conv2d(input_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(input_channels, num_anchors * 4, 3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        logits = self.cls_logits(out)
        bbox_reg = self.bbox_pred(out)
        return logits, bbox_reg

class Build(nn.Module):
    def __init__(self, num_classes, num_anchors = 9, init_weights = False):
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Backbone - 使用预训练ResNet50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后两层，输出C5特征

        # 简化FPN示例，只用C5层做示例
        self.fpn_layer = nn.Conv2d(2048, 256, 1)

        # 通常3个尺度*3个比例 (num_anchors)
        self.head = RetinaNetHead(256, num_anchors, num_classes)
        if init_weights:
            self._initialize_weights()
        

    def forward(self, x):
        c5 = self.body(x)          # backbone输出
        fpn_out = self.fpn_layer(c5)  # 简单FPN
        cls_logits, bbox_reg = self.head(fpn_out)
        N,AC,H,W = cls_logits.shape
        cls_logits = cls_logits.view(N,self.num_anchors,self.num_classes,H,W).permute(0,3,4,1,2).reshape(-1,self.num_classes)
        return cls_logits, bbox_reg
        
    def _initialize_weights(self):
        for i in self.modules():
            if isinstance(i,nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias,0)
                elif isinstance(i,nn.Linear):
                    nn.init.xavier_uniform_(i.weight)
                    nn.init.constant_(i.bias,0)
