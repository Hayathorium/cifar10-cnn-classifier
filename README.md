# CIFAR-10 CNN Classifier

This project is a complete deep learning pipeline for image classification using the CIFAR-10 dataset. It implements a custom Convolutional Neural Network (CNN) in PyTorch, featuring training with checkpointing, loss monitoring, and an easy-to-use inference script.

## Project Structure
- `model_CNN.py`: Defines the custom `CNN` architecture with convolutional, pooling, and linear layers.
- `train.py`: Handles dataset loading, model training, and automatic checkpoint saving/resuming.
- `predict.py`: Inference script to classify new images using your trained model.

## Features
- **Custom CNN**: A robust architecture tailored for 32x32 RGB images.
- **Checkpointing**: Training automatically resumes from `model.pth` if it exists.
- **Inference**: Easily classify custom images via the command line.

## Getting Started

### Prerequisites
- Python 3.x
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/)

### Training
To start training the model on the CIFAR-10 dataset, simply run:
```bash
python train.py

## Alternative Architectures

In addition to the custom CNN, this repository includes a **ResNet-34** implementation (`model_resnet34.py`). ResNet-34 is a deeper, more robust architecture that utilizes residual connections to mitigate the vanishing gradient problem in deeper networks.

* **Usage**:
To use this model, simply import the class in your training or inference scripts:
```python
from model_resnet34 import ResNet34

# Initialize for your specific dataset
model = ResNet34(in_channels=3, num_classes=10)
