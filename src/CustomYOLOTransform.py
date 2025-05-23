import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics.data.dataset import YOLODataset
from custom_transforms import get_custom_transform


class CustomYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        print("Attempting to initialize CustomYOLODataset with args:", args, "kwargs:", kwargs)
        try:
            super().__init__(*args, **kwargs)
            self.custom_transform = get_custom_transform()
            print("✅ CustomYOLODataset initialized successfully")
        except Exception as e:
            print(f"❌ CustomYOLODataset initialization failed: {e}")
            raise

    def __getitem__(self, index):
        print(f"Calling __getitem__ for index {index}")
        try:
            # Handle three return values from parent class
            img, labels, paths = super().__getitem__(index)
            print(f"Original image type: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
            print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}, Paths: {paths}")
            
            if isinstance(img, np.ndarray):
                if img.shape[0] in [1, 3]:  # CHW format
                    img = img.transpose(1, 2, 0)  # to HWC
                img = Image.fromarray(img.astype('uint8'), 'RGB')
            
            print(f"Applying custom transform to image {index}")
            img = self.custom_transform(img)
            print(f"Transformed image type: {type(img)}, shape: {img.shape}")
            
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
                
            # Return img and labels (paths optional, but not needed for training)
            return img, labels
        except Exception as e:
            print(f"❌ Error in __getitem__: {e}")
            raise