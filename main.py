import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Move the dataset class definition outside any function
class SimpleASLDataset(Dataset):
    def __init__(self, root_folder, transform=None, is_real_world=False, apply_special_processing=False):
        self.root_folder = root_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.is_real_world = is_real_world
        self.apply_special_processing = apply_special_processing
        
        # Identify classes
        self.classes = sorted([d for d in os.listdir(root_folder) 
                              if os.path.isdir(os.path.join(root_folder, d)) and d != "_cache"])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Store domain info for each image (0: synthetic, 1: real-world)
        self.domains = []
        
        for class_name in self.classes:
            class_folder = os.path.join(root_folder, class_name)
            if os.path.isdir(class_folder) and class_name != "_cache":
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                    # All images in this dataset are marked with the same domain
                    self.domains.append(1 if is_real_world else 0)
        
        print(f"Dataset contains {len(self.image_paths)} images across {len(self.classes)} classes")
        print(f"Type: {'Real-world' if is_real_world else 'Synthetic/Training'} data")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image {img_path}")
            
            # Apply enhanced preprocessing for real-world images if requested
            if self.is_real_world and self.apply_special_processing:
                img = enhance_hand_image(img)
            else:
                # Standard preprocessing for all images
                if len(img.shape) == 3:  # Color image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Normalize pixel values
                img = img.astype(np.float32) / 255.0
            
            # Apply transforms if provided
            if self.transform:
                img = self.transform(img)
            
            return img, torch.tensor(self.labels[idx])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and the label
            return torch.zeros((1, 224, 224)), torch.tensor(self.labels[idx])


# Combined dataset for mixing real-world and synthetic data
class MixedASLDataset(Dataset):
    def __init__(self, synthetic_dataset, real_world_dataset=None, real_world_weight=0.3):
        self.synthetic_dataset = synthetic_dataset
        self.real_world_dataset = real_world_dataset
        self.real_world_weight = real_world_weight  # Percentage of real-world samples to include
        
        # Combine datasets
        if real_world_dataset is not None:
            print(f"Creating mixed dataset with {len(synthetic_dataset)} synthetic and {len(real_world_dataset)} real-world images")
            
            # Use all synthetic images
            self.synthetic_indices = list(range(len(synthetic_dataset)))
            
            # Select a subset of real-world images (balancing classes if possible)
            real_world_class_indices = defaultdict(list)
            for i in range(len(real_world_dataset)):
                _, label = real_world_dataset[i]
                real_world_class_indices[label.item()].append(i)
            
            # Calculate how many real-world images to include per class
            num_classes = len(synthetic_dataset.classes)
            total_real_world = int(len(synthetic_dataset) * real_world_weight)
            per_class = max(1, total_real_world // num_classes)
            
            # Select balanced samples from each class
            self.real_world_indices = []
            for class_idx, indices in real_world_class_indices.items():
                # Randomly select up to per_class indices for this class
                if indices:
                    selected = random.sample(indices, min(per_class, len(indices)))
                    self.real_world_indices.extend(selected)
            
            print(f"Selected {len(self.real_world_indices)} real-world images for training")
        else:
            self.synthetic_indices = list(range(len(synthetic_dataset)))
            self.real_world_indices = []
            print("Using only synthetic dataset")
        
    def __len__(self):
        return len(self.synthetic_indices) + len(self.real_world_indices)
    
    def __getitem__(self, idx):
        # If idx is less than the number of synthetic samples, get from synthetic dataset
        if idx < len(self.synthetic_indices):
            return self.synthetic_dataset[self.synthetic_indices[idx]]
        
        # Otherwise, get from real-world dataset
        real_idx = idx - len(self.synthetic_indices)
        return self.real_world_dataset[self.real_world_indices[real_idx]]


# Improved CNN Model
class ImprovedASLModel(nn.Module):
    def __init__(self, num_classes, input_size=(224, 224)):
        super(ImprovedASLModel, self).__init__()
        
        # Calculate feature map sizes
        h, w = input_size
        h_out = h // 2 // 2 // 2  # After 3 max pooling layers of stride 2
        w_out = w // 2 // 2 // 2
        
        print(f"Feature map size after convolutions: {h_out}x{w_out}")
        fc_input_size = 64 * h_out * w_out
        print(f"Input features to fully connected layer: {fc_input_size}")
        
        # Improved CNN architecture
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Dropout for regularization
            nn.Dropout2d(0.2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Domain-Adaptive ASL Model (with domain classifier for adversarial training)
class DomainAdaptiveASLModel(nn.Module):
    def __init__(self, num_classes, input_size=(224, 224), alpha=1.0):
        super(DomainAdaptiveASLModel, self).__init__()
        
        # Alpha controls the importance of domain adaptation (0 = no adaptation)
        self.alpha = alpha
        
        # Calculate feature map sizes
        h, w = input_size
        h_out = h // 2 // 2 // 2  # After 3 max pooling layers of stride 2
        w_out = w // 2 // 2 // 2
        
        print(f"Feature map size after convolutions: {h_out}x{w_out}")
        fc_input_size = 64 * h_out * w_out
        print(f"Input features to fully connected layer: {fc_input_size}")
        
        # Feature extractor - same as the main model
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Class predictor (ASL letters)
        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Domain classifier (synthetic vs real)
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: synthetic vs real
        )
    
    def forward(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha
            
        # Extract features
        features = self.features(x)
        
        # Class prediction
        class_output = self.class_classifier(features)
        
        # For domain classification, we need to apply gradient reversal
        # Since PyTorch doesn't have a built-in GradientReversalLayer,
        # we'll implement it using a custom gradient scaling approach
        if self.training and alpha > 0:
            # Apply reverse gradient for domain adaptation during training
            reverse_features = GradientReversalFunction.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_features)
        else:
            domain_output = self.domain_classifier(features)
            
        return class_output, domain_output


# Gradient Reversal Layer for domain adaptation
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def enhance_hand_image(img_input, debug=False, output_folder="output_images"):
    """
    Enhance a hand image for better ASL recognition.
    
    Args:
        img_input: Input image as numpy array or path to image
        debug: Whether to show intermediate processing steps
        output_folder: Folder to save debug images to
    
    Returns:
        Processed image as numpy array
    """
    # Create output folder if it doesn't exist
    output_folder = ensure_output_folder(output_folder)
    
    # Load the image if a path is provided
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Could not read image at {img_input}")
    else:
        img = img_input.copy()
    
    # Store original for comparison
    original = img.copy()
    
    # Step 1: Resize if image is very large (keeping aspect ratio)
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size)
    
    # Step 2: Convert to different color spaces
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Step 3: Skin detection (using HSV and YCrCb)
    # HSV skin mask
    lower_hsv = np.array([0, 15, 30], dtype=np.uint8)
    upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    # YCrCb skin mask
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    # Step 4: Apply the mask to the original image
    skin = cv2.bitwise_and(img, img, mask=skin_mask)
    
    # Step 5: Find contours and keep only the largest ones (likely hands)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, use alternative approach
    if not contours:
        # Use thresholding on the grayscale image
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 1000
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    # If we have contours, find the largest one (likely the hand)
    if large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
        
        # Create a new mask with just the largest contour
        hand_mask = np.zeros_like(skin_mask)
        cv2.drawContours(hand_mask, [largest_contour], 0, 255, -1)
        
        # Get bounding rectangle and add some padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        # Crop to the hand region
        hand_region = img[y:y+h, x:x+w]
        
        # Make the background black (optional)
        mask_roi = hand_mask[y:y+h, x:x+w]
        hand_region_masked = cv2.bitwise_and(hand_region, hand_region, mask=mask_roi)
        
        # Convert to grayscale
        hand_gray = cv2.cvtColor(hand_region_masked, cv2.COLOR_BGR2GRAY)
        
        # Final preprocessing - normalize, equalize, etc.
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hand_equalized = clahe.apply(hand_gray)
        
        # Prepare final output
        output_image = hand_equalized
        
        # Resize to the model's expected size
        output_image = cv2.resize(output_image, (224, 224))
        
        # If debug mode, show all intermediate steps
        if debug:
            # Instead of displaying, save the debug visualization
            fig = plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            
            plt.subplot(2, 3, 2)
            plt.imshow(skin_mask, cmap='gray')
            plt.title('Skin Mask')
            
            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
            plt.title('Skin Segmentation')
            
            plt.subplot(2, 3, 4)
            plt.imshow(hand_mask, cmap='gray')
            plt.title('Hand Mask')
            
            plt.subplot(2, 3, 5)
            plt.imshow(hand_gray, cmap='gray')
            plt.title('Hand Grayscale')
            
            plt.subplot(2, 3, 6)
            plt.imshow(output_image, cmap='gray')
            plt.title('Final Processed')
            
            plt.tight_layout()
            
            # Get the filename if input was a path
            if isinstance(img_input, str):
                base_filename = os.path.basename(img_input)
            else:
                base_filename = "debug_image.png"
                
            # Save to output folder
            debug_image_path = os.path.join(output_folder, f"debug_steps_{base_filename}.png")
            plt.savefig(debug_image_path)
            plt.close(fig)  # Close the figure to avoid displaying it twice
            
        # Return the normalized image
        return output_image / 255.0
        
    else:
        # If no hand contour found, just return the grayscale image
        output_image = img_gray
        output_image = cv2.resize(output_image, (224, 224))
        return output_image / 255.0


def train_model_advanced(
    train_data_dir,
    real_world_data_dir,  # No longer optional
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    real_world_weight=0.3,
    apply_special_processing=True,
    output_model_path='asl_model_advanced_best.pth'
):
    """
    Advanced training function that always uses domain adaptation
    
    Args:
        train_data_dir: Directory with the main training dataset
        real_world_data_dir: Directory with real-world images (required)
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs to train
        learning_rate: Base learning rate
        real_world_weight: Weight of real-world examples in training (0-1)
        apply_special_processing: Whether to apply enhanced preprocessing to real-world images
        output_model_path: Path to save the best model
    
    Returns:
        Trained model
    """
    # Set multiprocessing start method to 'spawn' for Windows compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
        
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading training dataset from {train_data_dir}")
    print(f"Loading real-world dataset from {real_world_data_dir}")
    
    # Setup transforms
    # More aggressive augmentation for synthetic data
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),  # More rotation to match real-world variability
        transforms.RandomAffine(0, scale=(0.8, 1.2), translate=(0.1, 0.1)),  # More translation/scaling
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting variation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Less augmentation for real-world data (which is already variable)
    real_world_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),  # Less rotation
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = SimpleASLDataset(
        train_data_dir, 
        transform=train_transform,
        is_real_world=False
    )
    
    # Create real-world dataset
    if not os.path.exists(real_world_data_dir):
        raise ValueError(f"Real-world data directory {real_world_data_dir} does not exist")
    
    real_world_dataset = SimpleASLDataset(
        real_world_data_dir,
        transform=real_world_transform,
        is_real_world=True,
        apply_special_processing=apply_special_processing
    )
    
    # For domain adaptation, we need to keep track of domains
    combined_dataset = train_dataset  # We'll handle real-world separately
    
    # Split into train and validation
    dataset_size = len(combined_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create samplers
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0
    )
    
    # Create domain-specific loader for real-world data
    real_world_loader = DataLoader(
        real_world_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    num_classes = len(train_dataset.classes)
    print(f"Creating model with {num_classes} output classes")
    
    model = DomainAdaptiveASLModel(num_classes).to(device)
    print("Using domain-adaptive training model")
    
    # Criterion and optimizer
    class_criterion = FocalLoss(gamma=2.0)
    domain_criterion = nn.CrossEntropyLoss()
    
    # Optimizer for the entire model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Domain adaptation training loop
        # For each batch in the main dataset, also get a batch from the real-world dataset
        for i, (syn_data, real_data) in enumerate(zip(train_loader, 
                                                   # Cycle through real data if needed
                                                   cycle_dataloader(real_world_loader, len(train_loader))
                                                   )):
            # Synthetic data
            syn_images, syn_labels = syn_data
            syn_images, syn_labels = syn_images.to(device), syn_labels.to(device)
            
            # Domain labels: 0 for synthetic
            syn_domains = torch.zeros_like(syn_labels)
            
            # Real-world data
            real_images, real_labels = real_data
            real_images, real_labels = real_images.to(device), real_labels.to(device)
            
            # Domain labels: 1 for real-world
            real_domains = torch.ones_like(real_labels)
            
            # Combine data
            all_images = torch.cat([syn_images, real_images], dim=0)
            all_labels = torch.cat([syn_labels, real_labels], dim=0)
            all_domains = torch.cat([syn_domains, real_domains], dim=0)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # For domain adaptation, we get both class and domain predictions
            class_outputs, domain_outputs = model(all_images)
            
            # Class classification loss
            class_loss = class_criterion(class_outputs, all_labels)
            
            # Domain classification loss
            domain_loss = domain_criterion(domain_outputs, all_domains)
            
            # Total loss
            # Gradually increase domain adaptation importance
            alpha = min(1.0, (epoch + 1) / (num_epochs / 2))
            loss = class_loss + alpha * domain_loss
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = class_outputs.max(1)
            total += all_labels.size(0)
            correct += predicted.eq(all_labels).sum().item()
            
            # Print batch progress
            if (i+1) % 10 == 0:
                batch_acc = 100.0 * correct / total
                print(f"  Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.2f}%")
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs, _ = model(images)
                
                # Calculate loss
                loss = class_criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                correct_tensor = predicted.eq(labels)
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] += correct_tensor[i].item()
                    class_total[label] += 1
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        # Print statistics
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Print per-class accuracy
        print('  Per-class validation accuracy:')
        class_names = train_dataset.classes
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                print(f'    {class_names[i]}: {accuracy:.2f}%')
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f'  Saving model with improved validation accuracy: {val_acc:.2f}%')
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'classes': train_dataset.classes,
                'domain_adaptive': True
            }, output_model_path)
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    print(f'Training completed with best validation accuracy: {best_val_acc:.2f}%')
    return model

def test_model_with_preprocessing(
    image_path, 
    model_path="asl_model_advanced_best.pth", 
    data_dir=None,
    preprocessing_level='advanced',
    output_folder="output_images"
):
    """
    Test an image with enhanced preprocessing options.
    
    Args:
        image_path: Path to the test image
        model_path: Path to the model checkpoint
        data_dir: Path to dataset to get class names (if not in checkpoint)
        preprocessing_level: One of 'basic', 'standard', or 'advanced'
        output_folder: Folder to save output images to
    
    Returns:
        Dictionary with predictions and confidence
    """
    # Create output folder if it doesn't exist
    output_folder = ensure_output_folder(output_folder)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get class names
        if 'classes' in checkpoint:
            class_names = checkpoint['classes']
            num_classes = len(class_names)
            print(f"Loaded {num_classes} classes from checkpoint")
        elif data_dir:
            dataset = SimpleASLDataset(data_dir, transform=None)
            class_names = dataset.classes
            num_classes = len(class_names)
            print(f"Loaded {num_classes} classes from dataset directory")
        else:
            raise ValueError("No class names in checkpoint and no data_dir provided")
        
        # Check if domain adaptive
        is_domain_adaptive = checkpoint.get('domain_adaptive', False)
        
        # Create model
        if is_domain_adaptive:
            model = DomainAdaptiveASLModel(num_classes).to(device)
            print("Using domain-adaptive model")
        else:
            model = ImprovedASLModel(num_classes).to(device)
            print("Using standard model")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path} with validation accuracy: {checkpoint.get('val_acc', 'unknown')}%")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load and preprocess image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image '{image_path}'.")
            return None
        
        # Display input image info
        print(f"Input image shape: {img.shape}, dtype: {img.dtype}")
        
        # Get base image filename for outputs
        base_filename = os.path.basename(image_path)
        
        # Apply different preprocessing based on level
        if preprocessing_level == 'basic':
            # Basic preprocessing - just resize and normalize
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (224, 224))
            img_norm = img_resized.astype(np.float32) / 255.0
            
            # Save a copy for visualization
            output_path = os.path.join(output_folder, f"basic_preprocess_{base_filename}")
            cv2.imwrite(output_path, img_resized)
            preprocessed = img_norm
            
        elif preprocessing_level == 'standard':
            # Standard preprocessing with more normalization
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (224, 224))
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_resized)
            
            # Normalize
            img_norm = img_enhanced.astype(np.float32) / 255.0
            
            # Save a copy for visualization
            output_path = os.path.join(output_folder, f"standard_preprocess_{base_filename}")
            cv2.imwrite(output_path, img_enhanced)
            preprocessed = img_norm
            
        else:  # 'advanced'
            # Advanced preprocessing with hand detection and enhancement
            preprocessed = enhance_hand_image(img, debug=True, output_folder=output_folder)
            
            # Save a copy for visualization
            output_path = os.path.join(output_folder, f"advanced_preprocess_{base_filename}")
            cv2.imwrite(output_path, (preprocessed * 255).astype(np.uint8))
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(preprocessed).unsqueeze(0).to(device)
        print(f"Transformed tensor shape: {img_tensor.shape}")
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
    
    # Make prediction
    try:
        with torch.no_grad():
            if is_domain_adaptive:
                outputs, _ = model(img_tensor)
            else:
                outputs = model(img_tensor)
                
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Print results
        print(f"\nPredictions for {image_path} (using {preprocessing_level} preprocessing):")
        results = []
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            prob = top_probs[i].item() * 100
            print(f"  {class_names[idx]}: {prob:.2f}%")
            results.append({'label': class_names[idx], 'confidence': prob})
        
        # Create comparison visualization
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Preprocessed image
        plt.subplot(1, 2, 2)
        plt.imshow(preprocessed, cmap='gray')
        plt.title(f"Preprocessed ({preprocessing_level})\nTop: {results[0]['label']} ({results[0]['confidence']:.1f}%)")
        plt.axis('off')
        
        # Save to output folder
        output_path = os.path.join(output_folder, f"comparison_{base_filename}.png")
        plt.savefig(output_path)
        plt.show()
        
        return {
            'top_prediction': results[0],
            'all_predictions': results,
            'preprocessing': preprocessing_level
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def evaluate_preprocessing_methods(image_path, model_path="asl_model_advanced_best.pth", data_dir=None, output_folder="output_images"):
    """
    Compare different preprocessing methods on a single image.
    
    Args:
        image_path: Path to the test image
        model_path: Path to the model checkpoint
        data_dir: Path to dataset to get class names (if not in checkpoint)
        output_folder: Folder to save output images to
    """
    # Create output folder if it doesn't exist
    output_folder = ensure_output_folder(output_folder)
    
    # Try all preprocessing methods
    basic_result = test_model_with_preprocessing(
        image_path, model_path, data_dir, preprocessing_level='basic', output_folder=output_folder
    )
    
    standard_result = test_model_with_preprocessing(
        image_path, model_path, data_dir, preprocessing_level='standard', output_folder=output_folder
    )
    
    advanced_result = test_model_with_preprocessing(
        image_path, model_path, data_dir, preprocessing_level='advanced', output_folder=output_folder
    )
    
    # Combine results into a comparison table
    print("\n===== PREPROCESSING COMPARISON =====")
    print(f"{'Method':<10} {'Top Prediction':<10} {'Confidence':<10}")
    print("-" * 30)
    
    if basic_result:
        print(f"{'Basic':<10} {basic_result['top_prediction']['label']:<10} {basic_result['top_prediction']['confidence']:.2f}%")
    
    if standard_result:
        print(f"{'Standard':<10} {standard_result['top_prediction']['label']:<10} {standard_result['top_prediction']['confidence']:.2f}%")
    
    if advanced_result:
        print(f"{'Advanced':<10} {advanced_result['top_prediction']['label']:<10} {advanced_result['top_prediction']['confidence']:.2f}%")
    
    # Save comparison to a single visualization
    if basic_result and standard_result and advanced_result:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Basic: {basic_result['top_prediction']['label']}\n{basic_result['top_prediction']['confidence']:.1f}%")
        img = cv2.imread(os.path.join(output_folder, f"basic_preprocess_{os.path.basename(image_path)}"))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"Standard: {standard_result['top_prediction']['label']}\n{standard_result['top_prediction']['confidence']:.1f}%")
        img = cv2.imread(os.path.join(output_folder, f"standard_preprocess_{os.path.basename(image_path)}"))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Advanced: {advanced_result['top_prediction']['label']}\n{advanced_result['top_prediction']['confidence']:.1f}%")
        img = cv2.imread(os.path.join(output_folder, f"advanced_preprocess_{os.path.basename(image_path)}"))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"method_comparison_{os.path.basename(image_path)}.png"))
        plt.show()
    
    return {
        'basic': basic_result,
        'standard': standard_result,
        'advanced': advanced_result
    }



def debug_validation_performance(model_path, data_dir, num_samples=3, output_folder="output_images"):
    """
    Debug model performance on validation samples.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_dir: Path to the dataset directory
        num_samples: Number of random samples to test from each class
        output_folder: Folder to save output images to
    """
    # Create output folder if it doesn't exist
    output_folder = ensure_output_folder(output_folder)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get class names from checkpoint
        if 'classes' in checkpoint:
            class_names = checkpoint['classes']
        else:
            # Try to get class names from dataset
            dataset = SimpleASLDataset(data_dir, transform=None)
            class_names = dataset.classes
            
        num_classes = len(class_names)
        print(f"Found {num_classes} classes: {class_names}")
        
        # Create model
        is_domain_adaptive = checkpoint.get('domain_adaptive', False)
        if is_domain_adaptive:
            model = DomainAdaptiveASLModel(num_classes).to(device)
        else:
            model = ImprovedASLModel(num_classes).to(device)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path} with validation accuracy: {checkpoint.get('val_acc', 'unknown')}%")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create validation transform
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create dataset
    try:
        dataset = SimpleASLDataset(data_dir, transform=None)
        
        # Get examples of each class
        class_indices = defaultdict(list)
        for i in range(len(dataset)):
            _, label = dataset[i]
            class_indices[label.item()].append(i)
        
        # Test samples from each class
        for class_idx in range(num_classes):
            if class_idx not in class_indices or len(class_indices[class_idx]) == 0:
                print(f"Warning: No samples for class {class_names[class_idx]}")
                continue
                
            indices = class_indices[class_idx]
            samples_to_test = min(len(indices), num_samples)
            test_indices = np.random.choice(indices, samples_to_test, replace=False)
            
            print(f"\nTesting {samples_to_test} samples from class {class_names[class_idx]}")
            
            for i, idx in enumerate(test_indices):
                img_path = dataset.image_paths[idx]
                print(f"  Sample {i+1}: {os.path.basename(img_path)}")
                
                # Create a folder for this class if needed
                class_folder = os.path.join(output_folder, class_names[class_idx])
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                
                # Test with different preprocessing methods
                for method in ['basic', 'standard', 'advanced']:
                    result = test_model_with_preprocessing(
                        img_path, model_path, data_dir, 
                        preprocessing_level=method,
                        output_folder=class_folder
                    )
                    
                    if result:
                        top_pred = result['top_prediction']
                        is_correct = top_pred['label'] == class_names[class_idx]
                        print(f"    {method.capitalize()} preprocessing: {top_pred['label']} ({top_pred['confidence']:.2f}%) - {'✓' if is_correct else '✗'}")
    except Exception as e:
        print(f"Error during validation testing: {e}")
        return

# Helper function for cycling through a dataloader
def cycle_dataloader(dataloader, max_iterations):
    """
    Create an iterator that cycles through a dataloader for domain adaptation.
    
    Args:
        dataloader: The dataloader to cycle through
        max_iterations: Maximum number of iterations
    
    Returns:
        Generator that yields data from the dataloader
    """
    iterator = iter(dataloader)
    for i in range(max_iterations):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            data = next(iterator)
        yield data
        
        
# Create an output folder for saved images
def ensure_output_folder(output_folder="output_images"):
    """
    Create the output folder if it doesn't exist.
    
    Args:
        output_folder: Path to the output folder
    
    Returns:
        Path to the output folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    return output_folder



def main_menu():
    """
    Interactive menu for ASL recognition system
    """
    # Define default output folder
    output_folder = "output_images"
    
    print("\n===== ASL Recognition System =====")
    print("1. Train model")
    print("2. Test model on image")
    print("3. Compare preprocessing methods")
    print("4. Debug validation performance")
    print("5. Change output folder")
    print("6. Exit")
    
    choice = input("Enter your choice (1-6): ")
    
        
    if choice == '1':
        # Domain adaptation training
        train_data_dir = input("Enter training data directory [./asl_dataset]: ") or "./asl_dataset"
        real_world_data_dir = input("Enter real-world data directory (required): ")
        if not real_world_data_dir:
            print("Error: Real-world data directory is required for domain adaptation.")
            return
            
        batch_size = int(input("Enter batch size [32]: ") or "32")
        epochs = int(input("Enter number of epochs [50]: ") or "50")
        
        model = train_model_advanced(
            train_data_dir=train_data_dir,
            real_world_data_dir=real_world_data_dir,
            batch_size=batch_size,
            num_epochs=epochs,
        )
        
    elif choice == '2':
        # Test model
        image_path = input("Enter path to test image: ")
        if not os.path.exists(image_path):
            print(f"Error: Image '{image_path}' not found.")
            return
            
        model_path = input("Enter model path [asl_model_advanced_best.pth]: ") or "asl_model_advanced_best.pth"
        preprocess = input("Preprocessing level (basic/standard/advanced) [advanced]: ") or "advanced"
        
        result = test_model_with_preprocessing(
            image_path=image_path,
            model_path=model_path,
            preprocessing_level=preprocess,
            output_folder=output_folder
        )
        
    elif choice == '3':
        # Compare preprocessing methods
        image_path = input("Enter path to test image: ")
        if not os.path.exists(image_path):
            print(f"Error: Image '{image_path}' not found.")
            return
            
        model_path = input("Enter model path [asl_model_advanced_best.pth]: ") or "asl_model_advanced_best.pth"
        
        results = evaluate_preprocessing_methods(
            image_path=image_path,
            model_path=model_path,
            output_folder=output_folder
        )
        
    elif choice == '4':
        # Debug validation performance
        model_path = input("Enter model path [asl_model_advanced_best.pth]: ") or "asl_model_advanced_best.pth"
        data_dir = input("Enter data directory [./asl_dataset]: ") or "./asl_dataset"
        samples = int(input("Number of samples per class [3]: ") or "3")
        
        debug_validation_performance(
            model_path=model_path,
            data_dir=data_dir,
            num_samples=samples,
            output_folder=output_folder
        )
    
    elif choice == '5':
        # Change output folder
        new_folder = input(f"Enter new output folder path [{output_folder}]: ") or output_folder
        output_folder = new_folder
        ensure_output_folder(output_folder)
        print(f"Output folder changed to: {output_folder}")
        
    elif choice == '6':
        print("Exiting...")
        return
        
    else:
        print("Invalid choice, please try again.")
        
    # Return to menu after operation
    input("\nPress Enter to continue...")
    main_menu()


if __name__ == "__main__":
    main_menu()