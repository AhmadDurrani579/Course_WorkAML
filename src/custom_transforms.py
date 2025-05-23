import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics.data.dataset import YOLODataset

def get_custom_transform():
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()  # Minimal transform: just resize and convert to tensor
    ])