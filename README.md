# UNet for Image Segmentation with PyTorch
This project implements a **UNet** architecture using PyTorch for image segmentation tasks. The project consists of training and evaluating the model on a dataset of images and their corresponding ground truth segmentation masks. Further, it supports dropout regularization and compares results using different dropout probabilities.
## Introduction
The **UNet** is designed to take in images and output segmentation masks that label each pixel as belonging to either the foreground or background. This project implements **UNet** from scratch with PyTorch. The implemented model can be trained on a custom dataset, and evaluated using precision, recall, and F1 scores.
## Dataset Structure
The dataset should consist of two folders:
*	**images/**: Contains the input images (e.g., .jpg files).
*	**golds/**: Contains the corresponding ground truth segmentation masks (e.g., .png files). The filenames should match the format used in the code, where each ground truth mask's name corresponds to the input image (e.g., gold_<image_name>.png).
### The dataset is divided into:
*	**tr**: Training set
*	**val**: Validation set
*	**ts**: Test set
## Model Architecture
The architecture of this model is a modified U-Net. Key components include:
*	**Encoder**: Downsampling the input image using convolutional and max-pooling layers.
*	**Bridge**: A bottleneck layer that connects the encoder to the decoder.
*	**Decoder**: Upsampling the features and concatenating them with the corresponding encoder features using skip connections.
*	**Output Layer**: A 1x1 convolutional layer that generates the final segmentation map.
### There are two implementations of the UNet model:
1.	**Standard UNet**: Without dropout.
2.	**UNet with Dropout**: Dropout regularization can be introduced with various probabilities to reduce overfitting.
## Dropout Experiments
Different dropout probabilities (p=0.3, p=0.5, p=0.7) are applied to the UNet architecture to test the effects of regularization. These experiments aim to reduce overfitting and improve generalization on the test set.
