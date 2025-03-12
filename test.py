import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from main import ASLModel  # Import your trained model class

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure correct number of classes
num_classes = len(os.listdir("asl_dataset"))  # Use same class count as training

# Load trained model and move to GPU
model = ASLModel(num_classes).to(device)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
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

    # Apply edge detection
    img = cv2.Canny(img, 100, 200)

    # Resize to match training size
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize as done during training
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Move to GPU

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
        