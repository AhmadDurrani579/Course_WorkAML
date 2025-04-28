from ultralytics import YOLO
import albumentations as A
import cv2
import os
import torch
import numpy as np
from CustomAugmentations import CustomAugmentations
from yolo_loss import CustomLoss

# Augmentation config
AUG_CONFIG = {
    'horizontal_flip': True,
    'rotation': 30,
    'brightness': 0.5,
    'blur': True,
    'cutout': True,
    'affine': True,    #  enables scaling/shear
    'clahe': True      #  enables contrast enhancement
}


# Proper injection method


def inject_augmentations(trainer):
    original_dataset = trainer.train_loader.dataset
    augmentor = CustomAugmentations(AUG_CONFIG)
    
    # Preserve original method
    original_getitem = original_dataset.__getitem__
    
    def augmented_getitem(index):
        img, labels, paths, shapes = original_getitem(index)
        
        if original_dataset.augment:  # Only augment training data
            # Convert to Albumentations format
            img_np = img.permute(1, 2, 0).numpy()  # CHW → HWC
            boxes = labels[:, 1:].numpy()
            class_ids = labels[:, 0].int().numpy()  # Ensure class IDs are integers
            
            try:
                # Apply augmentations
                transformed = augmentor(
                    image=img_np,
                    bboxes=boxes,
                    class_labels=class_ids
                )
                
                # Convert back to YOLO format
                img = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()  # HWC → CHW
                labels = torch.cat([
                    torch.from_numpy(transformed['class_labels']).unsqueeze(1).float(),
                    torch.from_numpy(transformed['bboxes']).float()
                ], dim=1)
                
            except Exception as e:
                print(f"Augmentation failed for {paths}: {str(e)}")
                return img, labels, paths, shapes
        
        return img, labels, paths, shapes
    
    # Patch the dataset
    original_dataset.__getitem__ = augmented_getitem    

# Initialize and train
model = YOLO("yolov8n.pt")
model.model.loss = CustomLoss(model.model)  # Inject custom loss
model.add_callback("on_train_start", inject_augmentations)

results = model.train(
    data="dataset.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    augment=False,
    plots=True,
    
    name="aug_run_fixed"
)
