# Aircraft Detection and Classification using YOLO

This repository contains an implementation of a custom YOLOv8-based model for military aircraft detection and classification using PyTorch and Ultralytics.

## Key Features

- YOLOv8 baseline implementation for aircraft detection
- Custom head architectures including MLP and Transform variants
- Augmentation pipeline with advanced transforms
- Evaluation scripts for comparing model performance
- Web-based inference interface with interactive visualization

## Dataset Setup

This project uses the Military Aircraft Detection Dataset from Kaggle, containing 81 different aircraft classes.

## Installation and Setup

### 1. Clone the Repository

```bash
# Option 1: Download and unzip the folder
# Option 2: Clone the repository
git clone https://github.com/AhmadDurrani579/Course_WorkAML
```

### 2. Download the Dataset

Download the [Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) from Kaggle and unzip it into your project directory.

### 3. Configure the Dataset Path

Edit the `src/dataset.yaml` file to match your dataset location:

```yaml
train: /your/path/to/MilitaryAircraftDataset/images/train
val: /your/path/to/MilitaryAircraftDataset/images/val
test: /your/path/to/MilitaryAircraftDataset/images/test


```

### 4. Setup Virtual Environment

Create and activate a Python virtual environment in the `src` directory:

```bash
cd src
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics albumentations opencv-python torch numpy
```

## Running the Models

### Standard YOLOv8 (Default Head)

To run the base model with the default YOLOv8 head:

```bash
python main.py
```

### Running with Custom Heads (MLP or Transform)

To use custom heads, you need to modify the YOLOv8 configuration:

1. Navigate to the YOLOv8 configuration file:
```bash
cd venv/lib/python3.x/site-packages/ultralytics/cfg/models/v8/yolov8.yaml
```

2. Modify the head section:
   - Find the head section in the YAML file
   - Comment out or delete the existing head configuration
   - Add the appropriate custom head configuration:

#### For MLP Head:
```yaml
head:
  - [-1, 1, mlp_block.ModifiedYOLOv8Head, [nc, anchors, [ch0, ch1, ch2]]] 
```

#### For Transform Head:
```yaml
head:
  - [-1, 1, transform_block.YOLOv8HeadWithTransform, [nc, anchors, [ch0, ch1, ch2]]]
```

3. Save the file and return to the main directory

4. Edit `main.py` to select which head to use:
   - Find the following section at the end of `main.py`:
   ```python
   # Control which head to use
   use_mlp_head = False
   use_transform_head = True  # Set to True to use the transformation head, False otherwise
   ```
   - Set the appropriate variable to `True` based on your desired head

5. Run the model:
```bash
python main.py
```

