from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt', device='auto'):
        self.device = device
        print("Loading YOLOv8 detector...")
        self.model = YOLO(model_path)
        print(f"YOLOv8 detector loaded successfully on {device}")
        
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        
    def set_detection_params(self, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, image, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image from {image}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image_rgb = np.array(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            else:
                raise ValueError("Invalid image format")
        else:
            raise ValueError("Unsupported image type")
        
        results = self.model(
            image_rgb, 
            conf=conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': box.conf[0].cpu().numpy(),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_and_crop(self, image_path, conf_threshold=0.5):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detect(image_rgb, conf_threshold)
        
        cropped_regions = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop_img = image_rgb[y1:y2, x1:x2]
            if crop_img.size > 0:
                cropped_regions.append({
                    'image': crop_img,
                    'detection_info': det
                })
        
        return cropped_regions
