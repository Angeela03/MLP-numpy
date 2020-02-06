import numpy as np
import struct
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Define the hyperparameters. This is fixed now but gets changed to see how the model changes.
lr = 0.00001
epoch = 200
num_nodes = 100
activation = "relu"
np.random.seed(10)

def softmax(x):
    # Function for finding softmax values
    e = 10 ** -10
    denominator = (np.sum(np.exp(x), axis=1))
    denominator[denominator < e] = e
    denominator = denominator.reshape(-1, 1)
    o = np.exp(x) / denominator
    return o


def sigmoid_function(x):
    # Function for finding the sigmoid values
    return (1 / (1 + np.exp(-x)))


def relu(x):
    # Relu activation
    return np.maximum(0, x)


def relu_grad_all(x):
    # Finds the gradient of relu
    x[x < 0] = 0
    x[x >= 0] = 1
    return x


class MultiLayer:
    # The multilayer perceptron class
    def __init__(self, x, y,yy, verbose=False):
        self.x = x
        self.y = y
        self.w1 = np.random.normal(loc=0.0, scale=0.01, size=(x.shape[-1], num_nodes))
        self.b1 = np.zeros(num_nodes, dtype=float)
        self.w2 = np.random.normal(loc=0.0, scale=0.01, size=(num_nodes, 10))
        self.b2 = np.zeros(10, dtype=float)
        self.yy = yy

        if verbose:
            print(f'\n\nHidden layer weights: {self.w1.shape}\n{self.w1}\nHidden Layer bias: {self.b1.shape}\n{self.b1}')
            print(f'\n\nOutput layer weights: {self.w2.shape}\n{self.w2}\nOutput Layer bias: {self.b2.shape}\n{self.b2}')

    def forward(self, x, verbose=False):
        # Define the forward functions
        self.z1 = np.matmul(x, self.w1) + self.b1
        self.a1 = sigmoid_function(self.z1)
        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        y_hat = softmax(self.z2)
        if verbose:
            print(f'First layer y: {self.a1.shape} Max value = {np.max(self.a1)}\n{self.a1}')
            print(f'First layer activations: {self.z1.shape} Max value = {np.max(self.z1)}\n{self.z1}')
            print(f'\nOutput layer y: {self.a2.shape} Max value = {np.max(self.a2)}\n{self.a2}')
            print(f'\nOutput layer activation: {y_hat.shape} Max value = {np.max(y_hat)}\n{y_hat}')
        return y_hat

    def loss_function(self, y_hat):
        # The cross entropy loss function
        self.m = y_hat.shape[-1]
        self.n = y_hat.shape[0]
        sum_m = (np.sum(self.y * np.log(y_hat)))
        loss = (-1 / (self.m * self.n)) * sum_m
        return loss

    def backward(self, y_hat):

        # Calculate the derivative of w2 and b2
        dw2 = (1 / self.m) * np.matmul(self.a1.T, (y_hat - self.y))
        db2 = (1 / self.m) * np.sum((y_hat - self.y), axis=0)

        # Calculate the derivate of w1 and b1
        da1 = np.matmul((y_hat - self.y), self.w2.T)
        if activation == "sigmoid":
            dz1 = sigmoid_function(self.z1) * (1 - sigmoid_function(self.z1))
        else:
            dz1 = relu_grad_all(self.z1)
        dz1 = da1 * dz1
        dw1 = (1 / self.m) * np.matmul(self.x.T, dz1)
        db1 = (1 / self.m) * np.sum(dz1, axis=0)

        # Make changes in the weights and biases
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

    def calculate_accuracy(self, y, y_hat):
        # Calculate the accuracy
        y_hat_arg = np.argmax(y_hat, axis=1).reshape(-1,1)
        acc = accuracy_score(y, y_hat_arg)
        return acc

    def predict(self,x):
        pred = self.forward(x)
        return pred

    def fit(self):
        # Fitting the model
        list_loss = []
        list_epoch = []
        list_accuracy = []
        for i in range(epoch):
            print(f'\n\nEpoch: {i}')
            y_hat = self.forward(self.x)
            loss = self.loss_function(y_hat)
            print(f"Loss at {i} = {loss}")
            print((f"Accuracy at {i} = {self.calculate_accuracy(self.yy, y_hat)}"))
            list_epoch.append(i)
            list_loss.append(loss)
            list_accuracy.append(self.calculate_accuracy(self.yy, y_hat))
            self.backward(y_hat)

        # Plot the loss and accuracy
        plt.plot(list_epoch, list_loss)
        plt.show()

        plt.plot(list_epoch, list_accuracy)
        plt.show()


def main(train_x,train_y, test_x, test_y ):
    # Change the output into multiclass
    onehot_encoder = OneHotEncoder()
    train_y = np.array(train_y).reshape(len(train_y), 1)
    test_y = np.array(test_y).reshape(len(test_y), 1)
    onehot_train_y = onehot_encoder.fit_transform(train_y).toarray()
    onehot_test_y = onehot_encoder.transform(test_y).toarray()
    train_x = train_x.reshape(-1, 784)
    print('\nInput')
    print(train_x)
    # fit and predict
    test_x = test_x.reshape(-1, 784)
    classifier = MultiLayer(train_x, onehot_train_y, train_y)
    classifier.fit()
    predictions = classifier.predict(test_x)
    test_accuracy = classifier.calculate_accuracy(test_y, predictions)
    print("Test accuracy:", test_accuracy)
