import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

# Dataset class
class ASLDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_folder))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return torch.zeros((1, 64, 64)), torch.tensor(0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, torch.tensor(label)

# CNN Model
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self._to_linear = None
        self._compute_fc_input_size()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_fc_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)  # Change from (64,64) → (224,224)
            x = self.pool(self.relu(self.conv1(dummy_input)))
            x = self.pool(self.relu(self.conv2(x)))
            self._to_linear = x.view(1, -1).shape[1]  # Corrected size computation


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Function to train and save the model
def train_model():
    dataset = ASLDataset("./asl_dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = len(dataset.classes)
    model = ASLModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "asl_model.pth")
    print("Model trained and saved successfully!")

# Function to load and test the model on a new image
def test_model(image_path):
    dataset = ASLDataset("./asl_dataset")
    num_classes = len(dataset.classes)

    model = ASLModel(num_classes)
    model.load_state_dict(torch.load("asl_model.pth"))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(img_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        print(f"Predicted Label: {dataset.classes[predicted_idx]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        image_path = sys.argv[2]  # Pass image path as an argument
        test_model(image_path)
    else:
        train_model()  # Only trains if no "test" argument is given
