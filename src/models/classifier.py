import torch
import torch.nn as nn
from torchvision import models

class UrbanIssuesClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Use ResNet18 for simplicity and speed
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
