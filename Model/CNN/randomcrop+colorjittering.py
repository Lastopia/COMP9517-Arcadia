import random
import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from config import config

class RandomCrop:
    def __init__(self, scale=(0.6, 1.0), ratio=(0.8, 1.2)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, boxes, labels):
        if random.random() > 0.5:  
            return image, boxes, labels

        img_width, img_height = image.size
        boxes = boxes.copy()
        for _ in range(10):
            scale = random.uniform(self.scale[0], self.scale[1])
            ratio = random.uniform(self.ratio[0], self.ratio[1])

            crop_width = int(img_width * scale * ratio)
            crop_height = int(img_height * scale / ratio)

            crop_width = min(crop_width, img_width)
            crop_height = min(crop_height, img_height)

            x1 = random.randint(0, img_width - crop_width)
            y1 = random.randint(0, img_height - crop_height)
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            abs_boxes = []
            for box in boxes:
                x_center, y_center, width, height = box
                abs_x_center = x_center * img_width
                abs_y_center = y_center * img_height
                abs_width = width * img_width
                abs_height = height * img_height

                x_min = abs_x_center - abs_width / 2
                y_min = abs_y_center - abs_height / 2
                x_max = abs_x_center + abs_width / 2
                y_max = abs_y_center + abs_height / 2

                abs_boxes.append([x_min, y_min, x_max, y_max])

            valid_boxes = []
            valid_labels = []

            for i, (x_min, y_min, x_max, y_max) in enumerate(abs_boxes):
                inter_xmin = max(x_min, x1)
                inter_ymin = max(y_min, y1)
                inter_xmax = min(x_max, x2)
                inter_ymax = min(y_max, y2)

                if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                    new_x_min = (inter_xmin - x1) / crop_width
                    new_y_min = (inter_ymin - y1) / crop_height
                    new_x_max = (inter_xmax - x1) / crop_width
                    new_y_max = (inter_ymax - y1) / crop_height

                    new_x_center = (new_x_min + new_x_max) / 2
                    new_y_center = (new_y_min + new_y_max) / 2
                    new_width = new_x_max - new_x_min
                    new_height = new_y_max - new_y_min

                    if new_width > 0.02 and new_height > 0.02:  
                        valid_boxes.append([new_x_center, new_y_center, new_width, new_height])
                        valid_labels.append(labels[i])

            if valid_boxes:
                cropped_image = image.crop((x1, y1, x2, y2))
                return cropped_image, np.array(valid_boxes), np.array(valid_labels)

        return image, boxes, labels

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, boxes, labels):
        if random.random() > 0.5:  
            return image, boxes, labels

        transforms_to_apply = []

        if self.brightness > 0:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            transforms_to_apply.append(lambda img: F.adjust_brightness(img, brightness_factor))

        if self.contrast > 0:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            transforms_to_apply.append(lambda img: F.adjust_contrast(img, contrast_factor))

        if self.saturation > 0:
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            transforms_to_apply.append(lambda img: F.adjust_saturation(img, saturation_factor))

        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms_to_apply.append(lambda img: F.adjust_hue(img, hue_factor))

        random.shuffle(transforms_to_apply)

        jittered_image = image
        for transform in transforms_to_apply:
            jittered_image = transform(jittered_image)

        return jittered_image, boxes, labels

class AdvancedAugmentation:

    def __init__(self, config):
        self.config = config
        self.random_crop = RandomCrop(
            scale=config.AUGMENTATION['random_crop_scale'],
            ratio=config.AUGMENTATION['random_crop_ratio']
        )
        self.color_jitter = ColorJitter(
            brightness=config.AUGMENTATION['brightness'],
            contrast=config.AUGMENTATION['contrast'],
            saturation=config.AUGMENTATION['saturation'],
            hue=config.AUGMENTATION['hue']
        )

    def __call__(self, image, boxes, labels):
        original_image = image
        original_boxes = boxes.copy()
        original_labels = labels.copy()

        try:
            if self.config.AUGMENTATION['random_crop']:
                image, boxes, labels = self.random_crop(image, boxes, labels)

            if self.config.AUGMENTATION['color_jitter']:
                image, boxes, labels = self.color_jitter(image, boxes, labels)

            if len(boxes) == 0:
                return original_image, original_boxes, original_labels

            return image, boxes, labels

        except Exception as e:
            print(f"fail: {e}")
            return original_image, original_boxes, original_labels

def visualize_augmentations(image, boxes, labels, class_names, num_samples=5):
    from config import config

    aug = AdvancedAugmentation(config)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    img_width, img_height = image.size
    for box, label in zip(boxes, labels):
        x_center, y_center, width, height = box
        x_min = (x_center - width/2) * img_width
        y_min = (y_center - height/2) * img_height
        x_max = (x_center + width/2) * img_width
        y_max = (y_center + height/2) * img_height

        axes[0].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                      fill=False, edgecolor='red', linewidth=2))
        axes[0].text(x_min, y_min-5, class_names[int(label)],
                    bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7),
                    color='white', fontsize=8)

    for i in range(1, 6):
        aug_image, aug_boxes, aug_labels = aug(image.copy(), boxes.copy(), labels.copy())

        axes[i].imshow(aug_image)
        axes[i].set_title(f'Augmented Sample {i}')
        axes[i].axis('off')

        aug_width, aug_height = aug_image.size
        for box, label in zip(aug_boxes, aug_labels):
            x_center, y_center, width, height = box
            x_min = (x_center - width/2) * aug_width
            y_min = (y_center - height/2) * aug_height
            x_max = (x_center + width/2) * aug_width
            y_max = (y_center + height/2) * aug_height

            axes[i].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                          fill=False, edgecolor='blue', linewidth=2))
            axes[i].text(x_min, y_min-5, class_names[int(label)],
                        bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.7),
                        color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
