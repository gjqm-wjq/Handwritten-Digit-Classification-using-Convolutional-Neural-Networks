# Handwritten Digit Classification with Sequential CNN

## Overview

This repository contains code for a Sequential Convolutional Neural Network (CNN) model trained on the MNIST dataset for handwritten digit classification. The model architecture consists of convolutional layers, max-pooling, flattening, dense (fully connected) layers, dropout for regularization, and softmax activation for multi-class classification.

## Motivation

Handwritten digit classification is a fundamental problem in the field of computer vision and machine learning. It serves as a benchmark task for evaluating the effectiveness of various machine learning algorithms and deep learning architectures. This project aims to showcase the effectiveness of CNNs for handwritten digit classification tasks, leveraging the MNIST dataset, which is widely used for this purpose.

## Features

- Preprocessing: The dataset is preprocessed to normalize pixel values and reshape images for compatibility with the CNN model.
- Model Architecture: The CNN model consists of multiple convolutional layers followed by max-pooling, flattening, dense layers, dropout for regularization, and softmax activation for classification.
- Training and Evaluation: The model is trained on a training set and evaluated on a separate test set. Performance metrics such as accuracy and F1 score are calculated, and a confusion matrix is generated to assess the model's performance across different digit classes.
- Visualization: Training and validation accuracy and loss curves are plotted to visualize the learning dynamics of the model over epochs.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Results
- The trained Sequential CNN model achieves a test accuracy of approximately 99.47% and an F1 score that reflects this high level of performance.

- The confusion matrix reveals minimal misclassifications, indicating the model's proficiency in distinguishing between digit classes.

- Training and validation accuracy and loss curves demonstrate effective learning throughout the training process.

## License
- This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This project was inspired by the MNIST dataset and aims to showcase the effectiveness of CNNs for handwritten digit classification.

- Special thanks to the TensorFlow and scikit-learn communities for providing valuable resources and documentation.
