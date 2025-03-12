import cv2
import os
import torch
import numpy as np
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize as done during training
        ])

    def preprocess_image(self, img_path):
        """Load, preprocess, and transform an image from file."""
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' not found.")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image '{img_path}'.")
            return None

        img = cv2.resize(img, self.image_size)  # Resize to match model input size
        img_tensor = self.transform(img).unsqueeze(0)  # Convert to tensor
        return img_tensor

    def load_dataset_images(self, root_folder):
        """Load images and labels from dataset folder."""
        image_paths = []
        labels = []
        classes = sorted(os.listdir(root_folder))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for class_name in classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])

        return image_paths, labels, classes
