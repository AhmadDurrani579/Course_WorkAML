import torch
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# ✅ Load YOLO Model
yolo_model = YOLO("runs/detect/train6/weights/best.pt")
yolo_model.eval()



# ✅ Define CNN Model (Ensures Structure Matches Checkpoint)
class AircraftCNN(nn.Module):
    def __init__(self, num_classes):
        super(AircraftCNN, self).__init__()
        self.model = models.resnet18(weights=None)  # Match Training Model
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)  # Adjust Output Layer

    def forward(self, x):
        return self.model(x)

# ✅ Load checkpoint properly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_aircraft_model.pth", map_location=device)

# ✅ Use the correct number of classes from the saved model
num_classes = checkpoint["num_classes"]
cnn_model = AircraftCNN(num_classes=num_classes).to(device)

# ✅ Load model state dict properly
cnn_model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # Allow missing keys if needed
cnn_model.eval()  # Set model to evaluation mode

print(f"✅ CNN Model Loaded Successfully with {num_classes} classes!")

# ✅ Define Image Preprocessing (Same as CNN Training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Ensure Class Names Order is Correct
test_folder = "crop_resized/"
class_names = sorted(os.listdir(test_folder))  # Ensure consistent class ordering

y_true, y_pred_yolo, y_pred_cnn = [], [], []
yolo_times, cnn_times = [], []

# ✅ Process images in the test set
for class_name in class_names:
    class_folder = os.path.join(test_folder, class_name)

    if os.path.isdir(class_folder):
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)

            # ✅ Ground Truth
            y_true.append(class_name)

            # ✅ YOLO Prediction
            start_time = time.time()
            yolo_results = yolo_model(img_path, conf=0.25, iou=0.5)
            yolo_times.append(time.time() - start_time)

            if len(yolo_results[0].boxes) > 0:
                yolo_pred_label = yolo_results[0].names[int(yolo_results[0].boxes.cls[0])]
            else:
                yolo_pred_label = "Unknown"
            y_pred_yolo.append(yolo_pred_label)

            # ✅ CNN Prediction
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            start_time = time.time()
            with torch.no_grad():
                cnn_output = cnn_model(image)
                cnn_pred_class = torch.argmax(cnn_output).item()
            cnn_times.append(time.time() - start_time)

            # ✅ FIXED: Correct CNN Class Mapping
            y_pred_cnn.append(class_names[cnn_pred_class] if cnn_pred_class < len(class_names) else "Unknown")

# ✅ Convert to NumPy arrays
y_true = np.array(y_true)
y_pred_yolo = np.array(y_pred_yolo)
y_pred_cnn = np.array(y_pred_cnn)

# ✅ Compute Accuracy
yolo_accuracy = np.mean(y_true == y_pred_yolo)
cnn_accuracy = np.mean(y_true == y_pred_cnn)

print(f"\n📌 YOLO Accuracy: {yolo_accuracy:.4f}")
print(f"📌 CNN Accuracy: {cnn_accuracy:.4f}")

# ✅ Compare Inference Speed
print(f"\n⚡ Average YOLO Inference Time: {np.mean(yolo_times):.4f} sec/image")
print(f"⚡ Average CNN Inference Time: {np.mean(cnn_times):.4f} sec/image")

# ✅ Generate Confusion Matrices
cm_yolo = confusion_matrix(y_true, y_pred_yolo, labels=class_names)
cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=class_names)

# ✅ Plot YOLO Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm_yolo, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted (YOLO)")
plt.ylabel("True Labels")
plt.title("YOLO Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# ✅ Plot CNN Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm_cnn, annot=False, cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted (CNN)")
plt.ylabel("True Labels")
plt.title("CNN Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# ✅ Print Classification Reports
print("\n📊 YOLO Classification Report:")
print(classification_report(y_true, y_pred_yolo, labels=class_names))

print("\n📊 CNN Classification Report:")
print(classification_report(y_true, y_pred_cnn, labels=class_names))
