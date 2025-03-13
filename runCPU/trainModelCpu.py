import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import os
import numpy as np
import time
import torch.nn.functional as F
import importlib

# Use CPU device
device = torch.device("cpu")
print("yo")

# Dataset class with preloading for faster CPU training
class ASLDataset(Dataset):
    def __init__(self, root_folder, transform=None, preload_to_memory=True):
        self.root_folder = root_folder
        self.image_paths = []
        self.labels = []
        # Exclude folders starting with '_' (like _cache)
        self.classes = sorted([cls for cls in os.listdir(root_folder) if not cls.startswith('_')])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect image paths and labels
        for class_name in self.classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        # Default transform pipeline for 244x244 images (resized to 244x244)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((244, 244)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform
            
        self.preload_to_memory = preload_to_memory
        self.preloaded_data = None
        self.preloaded_labels = None
        if self.preload_to_memory:
            self._preload_to_memory()
    
    def _preload_to_memory(self):
        print("Preloading images to RAM...")
        batch_size = min(100, len(self.image_paths))
        sample_img = self._load_and_transform(self.image_paths[0])
        img_shape = sample_img.shape
        self.preloaded_data = torch.zeros((len(self.image_paths), *img_shape), dtype=torch.float32)
        self.preloaded_labels = torch.tensor(self.labels, dtype=torch.long)
        
        for i in range(0, len(self.image_paths), batch_size):
            end_idx = min(i + batch_size, len(self.image_paths))
            for j in range(i, end_idx):
                self.preloaded_data[j] = self._load_and_transform(self.image_paths[j])
        print(f"Preloaded {len(self.image_paths)} images to RAM.\n")
    
    def _load_and_transform(self, img_path):
        processImage = importlib.import_module("imageProcessing.removeImageNoise")
        img = processImage.detect_hand_edges(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            sample_shape = self.transform(np.zeros((100, 100, 3), dtype=np.uint8)).shape
            return torch.zeros(sample_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.preload_to_memory and self.preloaded_data is not None:
            return self.preloaded_data[idx], self.preloaded_labels[idx]
        # Fallback: on-the-fly processing
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        return self._load_and_transform(img_path), torch.tensor(label)

# ResNet-based model for ASL classification adapted for grayscale images
class ResNetASLModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetASLModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        # Modify first conv layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Optionally, initialize weights by averaging original weights over RGB channels.
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Function to train the model on CPU with CPU-optimized settings
def train_model_cpu(data_dir="./asl_dataset", batch_size=64, num_epochs=10, learning_rate=0.001):
    start_time = time.time()
    
    # Create dataset with preloading
    dataset = ASLDataset(data_dir, preload_to_memory=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use multiple workers for CPU DataLoader (adjust num_workers as appropriate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    
    num_classes = len(dataset.classes)
    model = ResNetASLModel(num_classes, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }, "asl_model.pth")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Function to test the model on a new image on CPU
def test_model(image_path, model_path="asl_model.pth", data_dir="./asl_dataset"):
    dataset = ASLDataset(data_dir, preload_to_memory=False)
    classes = dataset.classes
    num_classes = len(classes)
    
    model = ResNetASLModel(num_classes, pretrained=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val acc: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    start_time = time.time()
    processImage = importlib.import_module("imageProcessing.removeImageNoise")
    img = processImage.detect_skin(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
    
    top_probs, top_idxs = torch.topk(probabilities, 3)
    inference_time = (time.time() - start_time) * 1000  # ms
    print(f"Inference completed in {inference_time:.2f} ms")
    print(f"Predicted Label: {classes[top_idxs[0]]} with {top_probs[0]*100:.2f}% confidence")
    for i in range(len(top_idxs)):
        print(f"  {classes[top_idxs[i]]}: {top_probs[i]*100:.2f}%")
    
    return {
        'label': classes[top_idxs[0]],
        'confidence': top_probs[0].item()*100,
        'top_predictions': [(classes[top_idxs[i]], top_probs[i].item()*100) for i in range(len(top_idxs))],
        'inference_time_ms': inference_time
    }

# Uncomment the following lines to run training or testing on CPU
if __name__ == "__main__":
    # train_model_cpu()  # Uncomment to train
    test_model("test_image.jpg")  # Replace with your test image path
