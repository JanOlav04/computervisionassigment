import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
import torch.nn.functional as F
from torch.backends import cudnn
import torchvision.models as models
from colortest import process_image

# Enable cuDNN benchmarking and deterministic algorithms for better performance
cudnn.benchmark = True

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please use a GPU-enabled device.")

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")

# Advanced dataset with memory-mapped arrays for efficient data loading
class ASLDataset(Dataset):
    def __init__(self, root_folder, transform=None, preload_to_memory=True):
        self.root_folder = root_folder
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_folder))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder) and class_name != "_cache":
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        # Default transform pipeline with smaller image size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((244, 244)),  # Further reduced size
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform
            
        # Preload all images to GPU memory for maximum speed
        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_to_memory = preload_to_memory
        
        if self.preload_to_memory:
            self._preload_to_memory()
    
    def _preload_to_memory(self):
        """Preload all images to memory for faster training"""
        
        # Initialize tensors to hold all data
        batch_size = min(100, len(self.image_paths))  # Process in batches
        sample_img = self._load_and_transform(self.image_paths[0])
        img_shape = sample_img.shape
        
        # Allocate memory for all images and labels
        self.preloaded_data = torch.zeros((len(self.image_paths), *img_shape), dtype=torch.float32)
        self.preloaded_labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Process in batches to avoid memory issues
        total_batches = (len(self.image_paths) + batch_size - 1) // batch_size
        
        # Just print once at the beginning
        print(f"Preloading batches...")
        
        # Load all batches
        for i in range(0, len(self.image_paths), batch_size):
            end_idx = min(i + batch_size, len(self.image_paths))
            
            for j in range(i, end_idx):
                self.preloaded_data[j] = self._load_and_transform(self.image_paths[j])
        
        print(f"Preloaded {len(self.image_paths)} images to RAM. Total memory used: {self.preloaded_data.element_size() * self.preloaded_data.nelement() / (1024*1024):.2f} MB \n")

    def _load_and_transform(self, img_path):
        """Load and transform a single image"""
        img = cv2.imread(img_path)
        img = process_image(img)
        
        
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if img is None:
            print(f"Warning: Could not read {img_path}")
            # Create empty tensor with correct shape
            sample_shape = self.transform(np.zeros((100, 100, 3), dtype=np.uint8)).shape
            return torch.zeros(sample_shape)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload_to_memory and self.preloaded_data is not None:
            # Return preloaded data directly
            return self.preloaded_data[idx], self.preloaded_labels[idx]
        
        # Fallback to on-the-fly processing
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        return self._load_and_transform(img_path), torch.tensor(label)

class ResNetASLModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetASLModel, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 1-channel images instead of 3.
        # Note: One common strategy is to average or replicate the pretrained weights.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If you want to transfer weights from the pretrained model, you might want to initialize
        # the new conv1 weights by averaging over the RGB channels of the original weights.
        # For example:
        # orig_weights = models.resnet18(pretrained=True).conv1.weight.data
        # self.resnet.conv1.weight.data = orig_weights.mean(dim=1, keepdim=True)
        
        # Replace the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# Function to train with maximum GPU utilization
def train_model_gpu(data_dir="./asl_dataset", batch_size=64, num_epochs=10, learning_rate=0.001):
    """Train ASL model with optimized GPU utilization"""
    # Set PyTorch to release memory ASAP
    torch.cuda.empty_cache()
 
    # Time the whole process
    start_time = time.time()
    
    # Create dataset with preloading
    dataset = ASLDataset(data_dir, preload_to_memory=True)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 0 workers because data is already in memory
        pin_memory=False  # No need for pin_memory if already in RAM
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    num_classes = len(dataset.classes)
    
    # Create and move model to GPU
    model = ResNetASLModel(num_classes).to(device)
    
    # Use native AMP for mixed precision
    scaler = torch.amp.GradScaler(enabled=True, device='cuda')
    
    # Use SGD with momentum and weight decay
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Use cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    
    # Training loop with progress tracking
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Measure batch processing time
        batch_times = []
        data_times = []
        compute_times = []
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training...")
        
        batch_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            data_time = time.time() - batch_start
            data_times.append(data_time)
            
            # Move data to GPU
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Measure compute time
            compute_time = time.time() - batch_start - data_time
            compute_times.append(compute_time)
            
            # Only print occasionally to reduce output
            if batch_idx % 50 == 0:
                print(f"\rProgress: {batch_idx}/{len(train_loader)} batches | "
                      f"Loss: {train_loss/(batch_idx+1):.4f} | "
                      f"Acc: {100.0*correct/total:.2f}%", 
                      end='')
            
            # Measure batch time
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            batch_start = time.time()
            
            # Force GPU sync occasionally to prevent memory buildup
            if batch_idx % 50 == 0:
                torch.cuda.synchronize()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        print("\nValidating...")
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
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Calculate performance metrics
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_data_time = sum(data_times) / len(data_times)
        avg_compute_time = sum(compute_times) / len(compute_times)
        
        print(f"Performance: {1/avg_batch_time:.1f} batches/sec | "
              f"Data: {avg_data_time*1000:.1f}ms | "
              f"Compute: {avg_compute_time*1000:.1f}ms | "
              f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving best model with validation accuracy: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }, "asl_model.pth")
        
        # Force GPU memory cleanup between epochs
        torch.cuda.empty_cache()
    
    # Calculate and print total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Efficient inference function
def test_model(image_path, model_path="asl_model.pth", data_dir="./asl_dataset"):
    """Perform inference on a single image"""
    # Load dataset just for class names
    dataset = ASLDataset(data_dir, preload_to_memory=False)
    classes = dataset.classes
    num_classes = len(classes)
    
    # Create and load model
    model = ResNetASLModel(num_classes).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")
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
    
    # Time the inference
    start_time = time.time()
    
    # Process image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        
    # Get top predictions
    top_probs, top_idxs = torch.topk(probabilities, 3)
    
    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Print results
    print(f"Inference completed in {inference_time:.2f} ms")
    print(f"Predicted Label: {classes[top_idxs[0]]} with {top_probs[0]*100:.2f}% confidence")
    print("Top predictions:")
    for i in range(len(top_idxs)):
        print(f"  {classes[top_idxs[i]]}: {top_probs[i]*100:.2f}%")
    
    return {
        'label': classes[top_idxs[0]],
        'confidence': top_probs[0].item() * 100,
        'top_predictions': [(classes[top_idxs[i]], top_probs[i].item() * 100) for i in range(len(top_idxs))],
        'inference_time_ms': inference_time
    }