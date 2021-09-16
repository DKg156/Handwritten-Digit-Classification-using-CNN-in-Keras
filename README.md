# Handwritten-Digit-Classification-using-CNN-in-Keras
Project on classification of handwritten digits using Convolutional Neural Network (CNN) in Keras with the standard MNIST dataset .

The MNIST database is a popular database of handwritten digits that is commonly used in the field of Computer Vision and Deep Learning. Each image in the dataset is 28 x 28 square pixel. 60,000 images are used for training the model and 10,000 images are used to test it. In this project we will classify some handwritten digits and predict their labels using a CNN in Keras library. Make sure the deep learning library Keras is installed on your system.   

We have 10 classes of digits to predict. The Keras library provides an easy method for loading the MNIST dataset. After loading the dataset we reshape the data in a single channel and one hot encode the target values. We then scale and normalize our data. After this we start defining our Convolutional Network by pooling and flattening the layers. The model is then compiled and fitted on the training dataset.

We are using Matplotlib library to show the predicted results of some random images taken from the test dataset. The model predicts all the labels correctly indicating a high accuracy.
