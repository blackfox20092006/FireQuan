import os
import torch
import torch.nn as nn
from torchvision import models
import jax.numpy as jnp

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize standard ResNet18
        resnet = models.resnet18(pretrained=False) 
        
        # Re-initialize the FC layer to match the one it was fine-tuned on
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
        # Load custom fine-tuned weights if path exists
        if weights_path and os.path.exists(weights_path):
            resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[INFO] Loaded fine-tuned ResNet18 weights from {weights_path}")
        else:
            print(f"[WARN] Weights not found at {weights_path} or not provided.")
            print("[WARN] Using random non-pretrained initialization!")
            
        # Extract features by copying all children EXCEPT the final FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze all layers since it only serves as an extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
            
    def forward(self, x):
        """
        x: Input image tensor [B, C, H, W]
        Returns: Features tensor [B, 512]
        """
        # Ensure we never track gradients
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1) # Flattens [B, 512, 1, 1] -> [B, 512]
        return features

    def extract_features_jax(self, torch_images):
        """
        Helper method to extract features from a PyTorch batch and convert them 
        into JAX numpy arrays for compatibility with Flax/PennyLane downstream.
        
        Parameters:
        - torch_images: Batch of shape [B, 3, 224, 224] from PyTorch DataLoader
        """
        features_torch = self.forward(torch_images.to(self.device))
        return jnp.asarray(features_torch.cpu().numpy())
