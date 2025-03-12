import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from runCPU.trainModelCpu import ASLModel  # Import trained model class

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Get root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  

# Ensure correct number of classes
asl_dataset_path = os.path.join(root_dir, "asl_dataset")
if not os.path.exists(asl_dataset_path):
    raise FileNotFoundError(f"asl_dataset folder not found at: {asl_dataset_path}")

num_classes = len(os.listdir(asl_dataset_path))  # Use the same class count as training

# Load trained model
model_path = os.path.join(root_dir, "asl_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = ASLModel(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Load class labels (sorted to match training order)
class_labels = sorted(os.listdir(asl_dataset_path))

# Function to predict an image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    # Apply edge detection
    img = cv2.Canny(img, 100, 200)

    # Resize to match training size
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize as done during training
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Move to CPU

    with torch.no_grad():
        output = model(img_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    predicted_label = class_labels[predicted_idx]
    print(f"Predicted Sign: {predicted_label}")

# Example usage
# predict_image("test_image.jpg")  # Uncomment to test an image
