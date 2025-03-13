import torch
import cv2
import os
from torchvision import transforms
from runGPU.trainModelGpu import ResNetASLModel,  ASLDataset  # Import trained model class
from colortest import process_image
import importlib

device = torch.device("cuda")

def test_model(image_path, model_path="asl_model.pth", data_dir="asl_dataset"):
    """Perform inference on a single image"""
    # Load dataset just for class names
    dataset = ASLDataset(data_dir)
    classes = dataset.classes
    num_classes = len(classes)
    
    # Create and load model
    model = ResNetASLModel(num_classes).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # processImage = importlib.import_module("imageProcessing.removeImageNoise")
    # img = processImage.detect_hand_edges(image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (244, 244))
    img = process_image(img)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
    # Get the top prediction
    top_prob, top_idx = torch.max(probabilities, dim=0)
    
    # Print predicted label and confidence
    print(f"Predicted Label: {classes[top_idx]} with {top_prob.item() * 100:.2f}% confidence")
    
    return {
        'label': classes[top_idx],
        'confidence': top_prob.item() * 100
    }
