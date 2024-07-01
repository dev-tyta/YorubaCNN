import torch
import torch.nn as nn
import torch.nn.functional as F

class YorubaCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(YorubaCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, 5, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * (embed_dim), 128)
        self.fc2 = nn.Linear(128, num_classes)  # Adjust fc layer


    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        x = self.embedding(x)
        print(f"After embedding shape: {x.shape}")
        
        # Ensure x is 3D: [batch_size, embed_dim, seq_len]
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        print(f"After permute shape: {x.shape}")
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Dynamically calculate input size for the fully connected layer
        x = torch.flatten(x, 1)  # Flatten to prepare for the fully connected layer
        in_features = x.shape[1]
        self.fc = nn.Linear(in_features, self.fc.out_features)  # Adjust fc layer
        self.fc.to(x.device)  # Ensure fc is on the same device as input

        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Model initialization
vocab_size = 32000  # Number of tokens in the SentencePiece model
embed_dim = 256
num_classes = 9
model = YorubaCNN(vocab_size, embed_dim, num_classes)
model.eval()