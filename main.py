import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def fft_augmentation(data, max_segments=10, scale_ratio=2, alpha=0.5):
    # Shape of data: (n_samples, time_steps, n_features)
    n_samples, time_steps, n_features = data.shape

    # Ensure that n_features is divisible by max_segments
    if n_features % max_segments != 0:
        # Pad the data with zeros to make n_features divisible by max_segments
        padding_size = max_segments - (n_features % max_segments)
        data = torch.cat([data, torch.zeros(n_samples, time_steps, padding_size)], dim=-1)
        n_features = data.size(-1)  # Update n_features after padding

    # Reshaping the data into segments along the feature dimension
    segment_length = n_features // max_segments  # Now this is guaranteed to divide evenly
    data = data.view(n_samples, time_steps, max_segments, segment_length)

    augmented_data = []
    for i in range(n_samples):
        augmented_sample = data[i]  # (time_steps, max_segments, segment_length)

        # Shuffle the time steps for data augmentation
        augmented_sample = augmented_sample[torch.randperm(augmented_sample.size(0))]

        # Perform FFT on the shuffled data along the feature dimension
        data_fft = torch.fft.fft(augmented_sample, dim=-1)
        magnitude, phase = data_fft.abs(), data_fft.angle()

        # Apply random scaling and phase warping in the frequency domain
        scale_factor = torch.rand(time_steps, max_segments, 1) * scale_ratio  # Adjusting dimensions
        phase += (torch.rand(time_steps, max_segments, 1) - 0.5) * alpha  # Matching dimensions

        # Apply scaling and phase warping in the frequency domain
        augmented_sample = torch.fft.ifft(magnitude * scale_factor * torch.exp(1j * phase), dim=-1).real
        augmented_data.append(augmented_sample)

    # Stack the augmented data to return a tensor
    augmented_data = torch.stack(augmented_data)

    # Reshape augmented_data to 3D tensor (n_samples, time_steps * max_segments, segment_length)
    augmented_data = augmented_data.view(n_samples, -1, segment_length)  # Flatten time_steps and max_segments

    return augmented_data

# Load data function
def load_data(paths):
    data = {}
    for path in paths:
        key = path.split('/')[-1].split('.')[0]
        data[key] = torch.load(path, weights_only=True)
    return data

# Load HAR data for training (using val and test due to corrupted train file)
har_paths = ['/content/val.pt', '/content/test.pt']
har_data = load_data(har_paths)

# Load gesture dataset for evaluation
gesture_paths = ['/content/train_gesture.pt', '/content/val_gesture.pt', '/content/test_gesture.pt']
gesture_data = load_data(gesture_paths)

# Prepare DataLoader for HAR dataset (using val and test files)
train_data = torch.cat([har_data['val']['samples'], har_data['test']['samples']], dim=0)
train_labels = torch.cat([har_data['val']['labels'], har_data['test']['labels']], dim=0).long()

# Inspect the shape of the loaded train_data
print("Train data shape:", train_data.shape)

# Apply FFT-based augmentation
train_data_fft = fft_augmentation(train_data)

# Check the shape of the augmented data
print("Augmented train data shape:", train_data_fft.shape)

# Get the number of samples, time steps, and features
n_samples, time_steps, n_features = train_data.shape  # e.g., (4418, 3, 206)

# Ensure the data is in the shape (n_samples, time_steps, input_size)
train_data_fft = train_data_fft.view(n_samples, time_steps,21)  # Reshape data to match LSTM input

# Prepare DataLoader for gesture dataset (for evaluation)
eval_data = torch.cat([gesture_data['train_gesture']['samples'], gesture_data['val_gesture']['samples'], gesture_data['test_gesture']['samples']], dim=0)
eval_labels = torch.cat([gesture_data['train_gesture']['labels'], gesture_data['val_gesture']['labels'], gesture_data['test_gesture']['labels']], dim=0).long()
eval_loader = DataLoader(TensorDataset(eval_data, eval_labels), batch_size=64, shuffle=False)

# Set num_classes based on the HAR dataset
num_classes = len(torch.unique(train_labels))  # Assuming 6 classes in the dataset

# Multi-Layer LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=6):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)  # LSTM to extract features
        x = h_n[-1]  # Take the last hidden state as the feature vector
        return self.fc(x)

# Initialize the model with the correct number of input features and output classes
input_size = train_data.size(1)  # Assuming the data is time-series (samples, time-steps, features)
model = LSTMEncoder(input_size=input_size, num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare DataLoader for augmented train data
batch_size = 64  # Set a batch size as per requirement
train_dataset = TensorDataset(train_data_fft, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Pretraining on the HAR dataset (Self-Supervised Learning with Contrastive Loss)
for epoch in range(5):  # Pretraining epochs can be adjusted
    model.train()
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()

        # Apply FFT augmentation
        augmented_data = fft_augmentation(batch_data)

        # Forward pass
        outputs = model(augmented_data)  # Get representation vector
        loss = criterion(outputs, batch_labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f"Pretraining Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

# Fine-tuning on gesture dataset
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch_data, batch_labels in eval_loader:
        outputs = model(batch_data)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = 100 * correct / total
print(f"Model accuracy on gesture dataset: {accuracy:.2f}%")
