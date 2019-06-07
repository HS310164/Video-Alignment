# Video Alignment Process

This is Video alignment process.



# Requirement
- torch
- torchvision
- numpy
- opencv-python
- joblib
- tqdm

# Usage

## Feature Extract

Extract videos' feature by 2D CNN and 3D CNN.

This 3D CNN is a customized version of [Hara's work](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).

You can download pretrained model in his repository.

## Alignmnet

Align videos using extracted features.


You can measure the accuracy by `evaluate_labels.py`