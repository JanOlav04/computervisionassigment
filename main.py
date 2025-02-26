import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

# Custom Dataset Class for ASL
class ASLDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_folder))  # List of class (folder) names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # Map names to integers

        # Collect image paths and corresponding labels
        for class_name in self.classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder):  # Ensure it's a folder
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((64, 64)),  # Resize images
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return torch.zeros((1, 64, 64)), torch.tensor(0)  # Return empty tensor if failed

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct color format
        img = self.transform(img)

        return img, torch.tensor(label)

# Define the CNN Model
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Compute the correct input size for fc1
        self._to_linear = None
        self._compute_fc_input_size()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_fc_input_size(self):
        # Pass a dummy tensor through conv layers to get the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)  # Batch size 1, 1 channel, 64x64
            x = self.pool(self.relu(self.conv1(dummy_input)))
            x = self.pool(self.relu(self.conv2(x)))
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Dataset
dataset = ASLDataset("./asl_dataset")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize Model, Loss, Optimizer
num_classes = len(dataset.classes)
model = ASLModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")
