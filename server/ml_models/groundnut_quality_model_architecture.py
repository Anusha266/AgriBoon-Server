import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class GroundnutClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(GroundnutClassifier, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Transformer Encoder
        self.embed_dim = 64  # Matches output channels from conv3
        self.seq_len = 28 * 28  # Matches flattened spatial dimensions
        transformer_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, dim_feedforward=256, dropout=0.1)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=2)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN Feature Extraction
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 16, 112, 112)
        # #print(x)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 32, 56, 56)
        # #print(x)
        x = self.pool(self.relu(self.conv3(x)))  # (batch_size, 64, 28, 28)
        # #print(x)

        # Reshape for Transformer
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)
        #print(x)
        # Add Positional Encoding
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        #print(x)
        # Transformer Encoder
        x = self.transformer(x)  # (batch_size, seq_len, embed_dim)
        #print(x)
        # Flatten after transformer
        x = x.permute(0, 2, 1).contiguous().view(batch_size, -1)  # Flatten: (batch_size, 64 * 28 * 28)
        #print(x)
        # Fully Connected Layers for Classification
        x = self.relu(self.fc1(x))
        #print(x)
        x = self.dropout(x)
        #print(x)
        x = self.fc2(x)
        #print(x
        x = F.softmax(x)

        return x
