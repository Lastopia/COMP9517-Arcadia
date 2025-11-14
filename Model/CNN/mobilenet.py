class MobileNetClassifier:
    def __init__(self, num_classes=12, model_type='mobilenet_v3_small', pretrained=True):
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_type} classifier...")
        self.model = self._build_model(model_type, num_classes, pretrained)
        self.model = self.model.to(self.device)
        
        self.transform = self._get_transforms()
        self.class_names = [
            'aphid', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 
            'mites', 'mosquito', 'sawfly', 'stem_borer', 'stem_fly',
            'thrips', 'whitefly'
        ]
        
        print(f"MobileNet classifier loaded successfully on {self.device}")
    
    def _build_model(self, model_type, num_classes, pretrained):
        if model_type == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        elif model_type == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Unsupported image type")
        
        input_tensor = self.transform(pil_image)
        return input_tensor.unsqueeze(0)
    
    def predict(self, image, return_probabilities=False):
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_id = predicted.item()
        result = {
            'class_id': class_id,
            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
            'confidence': confidence.item(),
            'class_probabilities': probabilities.cpu().numpy()[0] if return_probabilities else None
        }
        
        return result
    
    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()
    
    def get_model_parameters(self):
        return self.model.parameters()
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', self.class_names)
        print(f"Model loaded from {filepath}")
