Learning TensorFlow 2.0 Notebooks
=================================

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/200px-TensorFlowLogo.svg.png"
title="TF-logo" />

## Running TensorFlow on Colab

### 1) Getting Started with TensorFlow 2.0: Training a Neural Network on MNIST

In this notebook, we train a shallow neural network to classify handwritten digits. We'll be using the tf.keras module which conveniently hides a lot of the complexity of neural networks. We train our neural network on the MNIST dataset, which is the "hello world" of Machine Learning and Deep Learning algorithms:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST" width="375" />

Our neural network's task is to classify images of handwritten digits, and with just a few lines of code we will be able to represent our neural network which looks similar to the one below:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/200px-Artificial_neural_network.svg.png" title="ANN" />

notebook: ([github](https://github.com/ccarpenterg/LearningTensorFlow2.0/blob/master/01_getting_started_with_tensorflow.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningTensorFlow2.0/blob/master/01_getting_started_with_tensorflow.ipynb))

### 2) Introduction to Convoutional Neural Networks and Deep Learning

If you're seeking a position in a computer vision related field, you should definitely know what a convolutional neural network is all about. Convolutional neural networks, Convnets or CNNs for short, are deep neural networks that automatically extract features from images through convolutional layers and then proceed to classify them through fully connected layers:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png" 
title="CNN" width="500" />

In this Colab notebook, we design a convolutional neural network from scratch and we train it on the MNIST dataset. Also we explore our convnet in terms of the numbers of parameters (weights and biases) to be trained and in terms of how the tensor dimensions change as it goes through the network.

notebook: ([github](https://github.com/ccarpenterg/LearningTensorFlow2.0/blob/master/02_introduction_to_convnets_and_deep_learning.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningTensorFlow2.0/blob/master/02_introduction_to_convnets_and_deep_learning.ipynb))

### 3) Plotting Accuracy and Loss for CNNs with TensorFlow

Part of the work that involves designing and training deep neural networks, consists in plotting the various parameters and metrics generated in the process of training. In this notebook we will design and train our Convnet from scratch, and will plot the training vs. test accuracy, and the training vs. test loss.

These are very important metrics, since they will show us how well is doing our neural network.

notebook: ([github](https://github.com/ccarpenterg/LearningTensorFlow2.0/blob/master/03_plotting_accuracy_loss_convnet.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningTensorFlow2.0/blob/master/03_plotting_accuracy_loss_convnet.ipynb))

### CIFAR-10: A More Challenging Dataset for CNNs

So far we have trained our neural networks on the MNIST dataset, and have achieved high acurracy rates for both the training and test datasets. Now we train our Convnet on the CIFAR-10 dataset, which contains 60,000 images of 32x32 pixels in color (3 channels) divided in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png" title="CIFAR-10" width="295" />

As we'll see in this notebook, the CIFAR-10 dataset will prove particularly challenging for our very basic Convnet, and from this point we'll start exploring the world of pretrained neural networks.

### Pretrained Convolutional Neural Networks

<img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16.png" title="VGG16" width="500" />
