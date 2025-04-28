from ultralytics import YOLO
from CustomAugmentations import CustomAugmentations
import torch
import numpy as np

class AugmentedYOLODataset:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.augmentor = CustomAugmentations(config)
        
    def __getitem__(self, index):
        img, labels, paths, shapes = self.dataset[index]
        
        if self.dataset.augment:
            img_np = img.permute(1, 2, 0).numpy()
            boxes = labels[:, 1:].numpy()
            class_ids = labels[:, 0].numpy()
            
            # Apply augmentations
            img_aug, boxes_aug, class_ids_aug = self.augmentor(
                img_np, boxes, class_ids
            )
            
            # Convert back to tensor
            labels = torch.cat([
                torch.from_numpy(np.array(class_ids_aug)).unsqueeze(1),
                torch.from_numpy(np.array(boxes_aug))
            ], dim=1)
            
            return img_aug, labels, paths, shapes
        return img, labels, paths, shapes

