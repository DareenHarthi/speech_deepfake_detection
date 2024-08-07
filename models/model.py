import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.layer2 = nn.Linear(128, 64)         # Second hidden layer
        self.output = nn.Linear(64, 1)           # Output layer
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x