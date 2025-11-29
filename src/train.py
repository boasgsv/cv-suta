import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import json
import random
import numpy as np

# Assuming these imports exist in your project structure
from src.data.dataset import UrbanIssuesDataset
from src.models.classifier import UrbanIssuesClassifier

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig):
    print(f"Training with config: \n{cfg}")
    
    # 1. SET SEED (Crucial for debugging spikes)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),        # Resize slightly larger
        transforms.RandomCrop((224, 224)),    # Randomly crop to target size
        transforms.RandomHorizontalFlip(p=0.5), # Flip left/right
        transforms.RandomRotation(degrees=15),  # Slight rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Change lighting conditions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = UrbanIssuesDataset(root_dir=cfg.data_dir, split="train", transform=train_transform)
    valid_dataset = UrbanIssuesDataset(root_dir=cfg.data_dir, split="valid", transform=valid_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # DataLoaders
    # 2. OPTIMIZATION: Use num_workers > 0 for faster data loading
    #    Use pin_memory=True if using GPU
    num_workers = cfg.get("num_workers", 4) 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    model = UrbanIssuesClassifier(num_classes=len(train_dataset.classes)).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training Loop
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": []
    }
    
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0 
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() # Accumulate into the separate variable
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        valid_loss = val_running_loss / len(valid_loader)
        valid_acc = 100 * correct / total

        scheduler.step(valid_loss)
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        
        # Save Best Model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model (Acc: {best_acc:.2f}%)")
            
    # Save final history
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("Saved training history to history.json")

if __name__ == "__main__":
    train()