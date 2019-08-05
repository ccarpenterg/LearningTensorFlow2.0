Learning TensorFlow 2.0 Notebooks
=================================

## Running TensorFlow on Colab

### 1) Getting Started in TensorFlow 2.0: Training a Neural Network on MNIST

In this notebook, we train a shallow neural network to classify handwritten digits. We'll be using the tf.keras module which conveniently hides a lot of the complexity of neural networks. We train our neural network on the MNIST dataset, which is the "hello world" of Machine Learning and Deep Learning algorithms:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST" width="375" />

With just a few lines of code we'll be able to represent a neural network like this one:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/200px-Artificial_neural_network.svg.png" title="ANN" />

### 2) Introduction to Convoutional Neural Networks and Deep Learning

If you're seeking a position in a computer vision related field, you should definitely know what a convolutional neural network is all about. Convolutional neural networks, Convnets or CNNs for short, are deep neural networks that automatically extract features from images through convolutional layers and then proceed to classify them through fully connected layers:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png" 
title="CNN" width="500" />

In this Colab notebook, we design a convolutional neural network from scratch and we train it on the MNIST dataset. Also we explore our convnet in terms of the numbers of parameters (weights and biases) to be trained and in terms of how the tensor dimensions change as it goes through the network.

### Plotting Accuracy and Loss for CNNs with TensorFlow


### CIFAR-10: A More Challenging Dataset for CNNs

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png" title="CIFAR-10" width="295" />

### Pretrained Convolutional Neural Networks

<img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16.png" title="VGG16" width="500" />
