import os
import torch
from torchvision import transforms
from PIL import Image
from model_CNN import CNN

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(3, 10).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

for filename in os.listdir("image"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join('image', filename)
        break

image = Image.open(path).convert("RGB")
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(x).argmax(1).item()

print("Prediction:", classes[pred])