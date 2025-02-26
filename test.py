import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from main import ASLModel  # Import your trained model class

# Ensure correct number of classes
num_classes = len(os.listdir("asl_dataset"))  # Use same class count as training

# Load trained model
model = ASLModel(num_classes)
model.load_state_dict(torch.load("asl_model.pth"))
model.eval()  # Set to evaluation mode

# Load class labels (sorted to match training order)
class_labels = sorted(os.listdir("asl_dataset"))

# Function to predict an image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    img = cv2.resize(img, (64, 64))  # Resize to match training size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize as done during training
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    predicted_label = class_labels[predicted_idx]
    print(f"Predicted Sign: {predicted_label}")

# Run prediction
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 test.py path/to/image.jpg")
    else:
        predict_image(sys.argv[1])
        