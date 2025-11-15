import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from config import config

class MobileNetFeatureYOLOv8:
    def __init__(self, config):
        self.config = config
        self.model = self._create_feature_model()
    
    def _create_feature_model(self):
        """创建YOLOv8模型，使用MobileNet进行特征增强"""
        try:
            # 使用预训练的YOLOv8模型
            model = YOLO('yolov8n.pt')
            print("Load the pre-trained YOLOv8 model.")
            
            # 修改类别数
            model.model.nc = config.NUM_CLASSES
            print(f"Change the number of categories to: {config.NUM_CLASSES}")
            
            # 添加MobileNet特征提取
            if config.FEATURE_FUSION:
                self._add_mobilenet_feature_extraction(model)
            
            return model
            
        except Exception as e:
            print(f"创建模型失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _add_mobilenet_feature_extraction(self, model):
        """添加MobileNet作为特征提取器"""
        # 加载预训练的MobileNet
        if self.config.MOBILENET_VERSION == "large":
            self.mobilenet = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if self.config.USE_PRETRAINED else None
            )
            feature_dim = 960  # MobileNetV3 Large特征维度
        else:
            self.mobilenet = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.config.USE_PRETRAINED else None
            )
            feature_dim = 576  # MobileNetV3 Small特征维度
        
        # 冻结MobileNet参数
        if self.config.FREEZE_MOBILENET:
            for param in self.mobilenet.parameters():
                param.requires_grad = False
            print("MobileNet特征提取器已冻结")
        
        # 创建特征融合模块
        class FeatureFusionModule(nn.Module):
            def __init__(self, mobilenet, feature_dim, yolo_channels):
                super().__init__()
                self.mobilenet = mobilenet
                # 特征适配层
                self.adapt_conv = nn.ModuleList([
                    nn.Conv2d(feature_dim, yolo_channels[0], 1),  # 适配P3
                    nn.Conv2d(feature_dim, yolo_channels[1], 1),  # 适配P4  
                    nn.Conv2d(feature_dim, yolo_channels[2], 1)   # 适配P5
                ])
                
            def forward(self, x, yolo_features):
                # 提取MobileNet特征
                mobilenet_feat = self.mobilenet.features(x)
                
                # 如果是单个tensor，直接返回（不融合）
                if not isinstance(yolo_features, (list, tuple)):
                    print(f"警告: YOLO输出是单个tensor，跳过特征融合")
                    return yolo_features
                
                fused_features = []
                for i, yolo_feat in enumerate(yolo_features):
                    if i >= len(self.adapt_conv):  # 防止索引越界
                        fused_features.append(yolo_feat)
                        continue
                        
                    # 适配MobileNet特征
                    adapted_feat = self.adapt_conv[i](mobilenet_feat)
                    
                    # 上采样到对应尺寸
                    upsampled_feat = nn.functional.interpolate(
                        adapted_feat, 
                        size=yolo_feat.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # 特征融合
                    fused_feat = yolo_feat + upsampled_feat
                    fused_features.append(fused_feat)
                
                return fused_features
        
        # 安全地获取YOLO通道数
        try:
            dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            with torch.no_grad():
                yolo_output = model(dummy_input)
            
            if isinstance(yolo_output, (list, tuple)):
                yolo_channels = [feat.shape[1] for feat in yolo_output]
                print(f"检测到YOLO输出包含 {len(yolo_channels)} 个特征图")
            else:
                # 使用合理的默认值
                yolo_channels = [128, 256, 512]
                print(f"使用默认通道数: {yolo_channels}")
                
        except Exception as e:
            print(f"获取YOLO通道数失败: {e}")
            yolo_channels = [128, 256, 512]
            print(f"使用备用通道数: {yolo_channels}")
        
        # 创建特征融合模块
        self.feature_fusion = FeatureFusionModule(
            self.mobilenet, feature_dim, yolo_channels
        )
        
        # 保存原始forward方法
        self.original_forward = model.model.forward
        
        # 创建新的forward方法
        def new_forward(x, *args, **kwargs):
            original_features = self.original_forward(x, *args, **kwargs)
            
            if self.config.FEATURE_FUSION:
                try:
                    fused_features = self.feature_fusion(x, original_features)
                    return fused_features
                except Exception as e:
                    print(f"特征融合失败，使用原始特征: {e}")
                    return original_features
            else:
                return original_features
        
        # 替换forward方法
        model.model.forward = new_forward
        print(f"MobileNetV3 {self.config.MOBILENET_VERSION} 特征融合模块已添加")
    
    def get_model(self):
        return self.model
