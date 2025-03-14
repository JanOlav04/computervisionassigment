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

# Main Menu
def main_menu():
    while True:
        print("\n===== ASL Model Menu =====")
        print("1. Train model on CPU")
        print("2. Test model on CPU")
        print("3. Train model on GPU")
        print("4. Test model on GPU")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            train_model_cpu()
        
        elif choice == "2":
            image_path = input("Enter the path of the image you want to test: ")
            predict_image(image_path)
        
        elif choice == "3":
            train_model_gpu()
        
        elif choice == "4":
            
            test_model_gpu("testSet/nikoT.jpg")
        
        elif choice == "5":
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1 and 5.")

# ASL Dataset Class
class ASLDataset(Dataset):
    def __init__(self, root_folder, transform=None, preload_to_memory=True):
        self.root_folder = root_folder
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

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.RandomResizedCrop(size=(244, 244), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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
        batch_size = min(100, len(self.image_paths))
        sample_img = self._load_and_transform(self.image_paths[0])
        img_shape = sample_img.shape
        
        self.preloaded_data = torch.zeros((len(self.image_paths), *img_shape), dtype=torch.float32)
        self.preloaded_labels = torch.tensor(self.labels, dtype=torch.long)
        
        for i in range(0, len(self.image_paths), batch_size):
            end_idx = min(i + batch_size, len(self.image_paths))
            for j in range(i, end_idx):
                self.preloaded_data[j] = self._load_and_transform(self.image_paths[j])
    
    # In the ASLDataset class:
    def _load_and_transform(self, img_path):
        try:
            img = detect_hand_edges(img_path)
            if img is None:
                return torch.zeros((1, 224, 224))
            
            # Convert single-channel to 3-channel if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return self.transform(img)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros((1, 224, 224))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.preload_to_memory and self.preloaded_data is not None:
            return self.preloaded_data[idx], self.preloaded_labels[idx]
        return self._load_and_transform(self.image_paths[idx]), torch.tensor(self.labels[idx])

# ResNet ASL Model
class ResNetASLModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetASLModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Train Model on CPU
def train_model_cpu(data_dir="./asl_dataset", batch_size=32, num_epochs=10, learning_rate=0.001):
    device = torch.device("cpu")
    dataset = ASLDataset(data_dir, preload_to_memory=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=False)
    
    num_classes = len(dataset.classes)
    model = ResNetASLModel(num_classes, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        
        train_acc = 100.0 * correct / total
        
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
        val_acc = 100.0 * correct / total
        
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, "asl_model_best.pth")
    
    print(f"Training completed in {(time.time() - start_time):.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Test Model on CPU
def predict_image(image_path, confidence_threshold=0.7):
    device = torch.device("cpu")
    dataset = ASLDataset("./asl_dataset", preload_to_memory=False)
    class_labels = sorted(os.listdir("./asl_dataset"))
    
    model = ResNetASLModel(len(class_labels), pretrained=False).to(device)
    checkpoint = torch.load("asl_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    img = detect_hand_edges(image_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
    
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    top_predictions = [(class_labels[idx], prob) for idx, prob in zip(top_indices, top_probs)]
    
    print("Top 3 predictions:")
    for label, prob in top_predictions:
        print(f"{label}: {prob*100:.2f}%")
    
    if confidence >= confidence_threshold:
        predicted_label = class_labels[predicted_idx]
        print(f"Predicted Sign: {predicted_label} (Confidence: {confidence*100:.2f}%)")
        return predicted_label, confidence
    else:
        print(f"Prediction confidence ({confidence*100:.2f}%) below threshold ({confidence_threshold*100:.2f}%)")
        return None, confidence

# Train Model on GPU
def train_model_gpu(data_dir="./asl_dataset", batch_size=64, num_epochs=20, learning_rate=0.001):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU-enabled device.")
    
    start_time = time.time()
    
    device = torch.device("cuda")
    dataset = ASLDataset(data_dir, preload_to_memory=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    num_classes = len(dataset.classes)
    model = ResNetASLModel(num_classes).to(device)
    
    scaler = torch.amp.GradScaler(enabled=True, device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Debugging: print the structure of a single batch to check what's being returned
        for batch_idx, batch in enumerate(train_loader):

            # Ensure the batch is correctly unpacked
            if len(batch) == 2:
                images, labels = batch
            else:
                # You can print more details here to inspect if the batch structure is different
                print("Unexpected batch structure")
                continue
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
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
    
    print(f"Training completed in {(time.time() - start_time):.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


# Test Model on GPU
def test_model_gpu(image_path, model_path="asl_model.pth", data_dir="./asl_dataset"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU-enabled device.")
    
    device = torch.device("cuda")
    dataset = ASLDataset(data_dir, preload_to_memory=False)
    classes = dataset.classes
    num_classes = len(classes)
    
    model = ResNetASLModel(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img = detect_hand_edges(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    top_prob, top_idx = torch.max(probabilities, dim=0)
    print(f"Predicted Label: {classes[top_idx]} with {top_prob.item() * 100:.2f}% confidence")
    
    return {
        'label': classes[top_idx],
        'confidence': top_prob.item() * 100
    }

# Image Processing Function
def detect_hand_edges(image_path, output_path=None, debug=False):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (244, 244))
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None
    
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(skin_mask)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            cv2.drawContours(hand_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    hand_region = cv2.bitwise_and(img, img, mask=hand_mask)
    gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_hand, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    final_edges = cv2.bitwise_and(edges, edges, mask=hand_mask)
    final_edges = cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR)
    return final_edges

if __name__ == "__main__":
    main_menu()