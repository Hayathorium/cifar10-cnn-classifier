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
