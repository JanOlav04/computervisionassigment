import torch
import cv2
import os
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from runCPU.trainModelCpu import ResNetASLModel  # Import trained model class
import importlib

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

model = ResNetASLModel(num_classes).to(device)

# Fix the model loading issue by extracting just the model_state_dict
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

# Load class labels (sorted to match training order)
class_labels = sorted(os.listdir(asl_dataset_path))

# Function to predict an image with confidence threshold
def predict_image(image_path, confidence_threshold=0.7):
    """
    Predict ASL sign from an image with a specified confidence threshold.
    
    Args:
        image_path (str): Path to the image file
        confidence_threshold (float): Minimum confidence required to make a prediction (0-1)
        
    Returns:
        tuple: (predicted_label, confidence) or (None, 0) if confidence is below threshold
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None, 0

    processImage = importlib.import_module("imageProcessing.removeImageNoise")
    img = processImage.detect_hand_edges(image_path)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Move to CPU

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()

    # Get top 3 predictions with their confidence scores
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    top_predictions = [(class_labels[idx], prob) for idx, prob in zip(top_indices, top_probs)]
    
    # Print all predictions with confidence scores
    print("Top 3 predictions:")
    for label, prob in top_predictions:
        print(f"{label}: {prob*100:.2f}%")
    
    # Check if the confidence meets the threshold
    if confidence >= confidence_threshold:
        predicted_label = class_labels[predicted_idx]
        print(f"Predicted Sign: {predicted_label} (Confidence: {confidence*100:.2f}%)")
        return predicted_label, confidence
    else:
        print(f"Prediction confidence ({confidence*100:.2f}%) below threshold ({confidence_threshold*100:.2f}%)")
        print("Consider retaking the image or adjusting the threshold.")
        return None, confidence

# Example usage
# predict_image("test_image.jpg", confidence_threshold=0.70)  # Prediction with 70% confidence threshold

