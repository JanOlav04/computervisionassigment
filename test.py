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
from torch.backends import cudnn



class SimpleASLModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleASLModel, self).__init__()
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Dropout for regularization
            nn.Dropout2d(0.3),
        )
        
        # Calculate input size for the fully connected layer
        # Input: 224x224 -> After 4 max pooling layers: 14x14
        fc_input_size = 256 * 14 * 14
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_model_gpu_simple(data_dir="./asl_dataset", val_dir="./valSet", batch_size=32, num_epochs=50, learning_rate=0.0001):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU-enabled device.")
    
    start_time = time.time()
    device = torch.device("cuda")
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create a simpler dataset class that doesn't use edge detection
    class SimpleASLDataset(Dataset):
        def __init__(self, root_folder, transform=None):
            self.root_folder = root_folder
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.classes = sorted(os.listdir(root_folder))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for class_name in self.classes:
                class_folder = os.path.join(root_folder, class_name)
                if os.path.isdir(class_folder) and class_name != "_cache":
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                img = cv2.imread(img_path)
                # Convert to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if self.transform:
                    img = self.transform(img)
                
                return img, torch.tensor(self.labels[idx])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image and the label
                return torch.zeros((1, 224, 224)), torch.tensor(self.labels[idx])
    
    # Load datasets
    train_dataset = SimpleASLDataset(data_dir, transform=transform)
    
    # Split into train and validation sets
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = SimpleASLModel(num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
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
        
        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f'  Saving model with improved validation accuracy: {val_acc:.2f}%')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, 'asl_model_simple_best.pth')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    print(f'Training completed in {(time.time() - start_time):.2f} seconds')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    return model


train_model_gpu_simple()