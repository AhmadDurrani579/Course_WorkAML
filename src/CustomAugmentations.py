import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

class CustomAugmentations:
    def __init__(self, config):
        """
        Initialize with augmentation configuration.
        
        Args:
            config (dict): Dictionary containing augmentation settings.
                Example:
                {
                    'horizontal_flip': True,
                    'rotation': 30,
                    'brightness': 0.5,
                    'blur': True,
                    'cutout': True,
                    'affine': True,
                    'clahe': True
                }
        """
        self.config = config
        self.transform = self._build_pipeline()

    def _build_pipeline(self):
        """Build the augmentation pipeline based on config."""
        return A.Compose(
            transforms=self._get_transforms(),
            bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.4,  # Discard boxes with <40% visibility
                label_fields=['class_labels']
            )
        )

    def _get_transforms(self):
        """Generate list of transforms based on config."""
        transforms = []
        
        # --- Geometric Transforms ---
        if self.config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
            
        if self.config.get('rotation'):
            transforms.append(A.Rotate(limit=self.config['rotation'], p=0.5))
            
        if self.config.get('affine', False):
            transforms.append(A.Affine(
                scale=(0.8, 1.2),  # 80-120% scaling
                shear=15,          # ±15 degrees
                p=0.5
            ))
        
        # --- Color Transforms ---
        if self.config.get('brightness'):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.config['brightness'],
                contrast_limit=0.1,  # Minimal contrast change
                p=0.5
            ))
            
        if self.config.get('clahe', False):
            transforms.append(A.CLAHE(p=0.5))
        
        # --- Advanced Augmentations ---
        if self.config.get('blur', False):
            transforms.append(A.GaussianBlur(
                blur_limit=(3, 7),  # Kernel size range
                p=0.2
            ))
            
        if self.config.get('cutout', False):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,  # Black patches
                p=0.5
            ))
        
        # --- Mandatory Transforms ---
        transforms.extend([
            A.Resize(640, 640),  # YOLO standard size
            ToTensorV2()          # Convert to PyTorch tensor
        ])
        
        return transforms

    def __call__(self, image, bboxes, class_labels):
        """
        Apply augmentations to image and bounding boxes.
        
        Args:
            image: Input image (numpy array, HWC format)
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class IDs
        
        Returns:
            Tuple: (augmented_image, augmented_bboxes, augmented_class_labels)
        """
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']