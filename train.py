import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_CNN import CNN

# data
trainloader = DataLoader(
    datasets.CIFAR10("./data", download=True, transform=transforms.ToTensor()),
    batch_size=64,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(3, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

path = "model.pth"

# resume
if os.path.exists(path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print("Resumed training")

# training
for epoch in range(50):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# save ONLY at the end
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, path)

print("Training finished and model saved.")