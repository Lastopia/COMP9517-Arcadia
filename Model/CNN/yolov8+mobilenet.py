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
        try:
            model = YOLO('yolov8n.pt')
            print("Load the pre-trained YOLOv8 model.")
            
            model.model.nc = config.NUM_CLASSES
            print(f"Change the number of categories to: {config.NUM_CLASSES}")
            
            if config.FEATURE_FUSION:
                self._add_mobilenet_feature_extraction(model)
            
            return model
            
        except Exception as e:
            print(f"fail: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _add_mobilenet_feature_extraction(self, model):
        if self.config.MOBILENET_VERSION == "large":
            self.mobilenet = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if self.config.USE_PRETRAINED else None
            )
            feature_dim = 960  
        else:
            self.mobilenet = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.config.USE_PRETRAINED else None
            )
            feature_dim = 576  
        
        if self.config.FREEZE_MOBILENET:
            for param in self.mobilenet.parameters():
                param.requires_grad = False
            print("MobileNet freeze")
        
        class FeatureFusionModule(nn.Module):
            def __init__(self, mobilenet, feature_dim, yolo_channels):
                super().__init__()
                self.mobilenet = mobilenet
                self.adapt_conv = nn.ModuleList([
                    nn.Conv2d(feature_dim, yolo_channels[0], 1),  
                    nn.Conv2d(feature_dim, yolo_channels[1], 1),    
                    nn.Conv2d(feature_dim, yolo_channels[2], 1)   
                ])
                
            def forward(self, x, yolo_features):
                mobilenet_feat = self.mobilenet.features(x)
                
                if not isinstance(yolo_features, (list, tuple)):
                    print(f"single tensor")
                    return yolo_features
                
                fused_features = []
                for i, yolo_feat in enumerate(yolo_features):
                    if i >= len(self.adapt_conv):  
                        fused_features.append(yolo_feat)
                        continue
                        
                    adapted_feat = self.adapt_conv[i](mobilenet_feat)
                    
                    upsampled_feat = nn.functional.interpolate(
                        adapted_feat, 
                        size=yolo_feat.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    fused_feat = yolo_feat + upsampled_feat
                    fused_features.append(fused_feat)
                
                return fused_features
        
        try:
            dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            with torch.no_grad():
                yolo_output = model(dummy_input)
            
            if isinstance(yolo_output, (list, tuple)):
                yolo_channels = [feat.shape[1] for feat in yolo_output]
                print(f" {len(yolo_channels)} features")
            else:
                yolo_channels = [128, 256, 512]
                print(f"channel: {yolo_channels}")
                
        except Exception as e:
            print(f"fail: {e}")
            yolo_channels = [128, 256, 512]
            print(f"channel: {yolo_channels}")
        
        self.feature_fusion = FeatureFusionModule(
            self.mobilenet, feature_dim, yolo_channels
        )
        
        self.original_forward = model.model.forward
        
        def new_forward(x, *args, **kwargs):
            original_features = self.original_forward(x, *args, **kwargs)
            
            if self.config.FEATURE_FUSION:
                try:
                    fused_features = self.feature_fusion(x, original_features)
                    return fused_features
                except Exception as e:
                    print(f"fail: {e}")
                    return original_features
            else:
                return original_features
        
        model.model.forward = new_forward
        print(f"MobileNetV3 {self.config.MOBILENET_VERSION}")
    
    def get_model(self):
        return self.model
