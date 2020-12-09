# HW5 Report

> Author: Jiayi Liu

This is an individual work. The implementation is in the `neural_network.py` file.

## 1. Experiment Result

The prediction and accuracy of test data from one of the experiments are as follows.

```python
Prediction:
[1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0
 1 1 0 0 0 0 0 0 0]
Accuracy:  0.963855421686747
```

## 2. Implementation Detail

 There are three main classes implemented: class `NeuralNetwork`, class `Linear`, and class `Sigmoid`.

### 2.1 Sigmoid

One of the challenges is the simple implementation of sigmoid function will easily lead to overflow. To prevent the overflow, a simple trick was used that two formula applied for positive and negative values, respectively.

```python
class Sigmoid:
    def forward(self, x):
        # Prevent overflow.
        self.out = np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))
        return self.out
```

### 2.2 Linear

Since each perceptron simulates a linear function as `y = Wx+b`, a linear layer was encapsulated in this `Linear` class. To make the computation more efficiently, a column of ones was concatenated to the last column of the input so that the matrix multiplication can compute the bias together.

### 2.3 NeuralNetwork

In the constructor, the network architecture and parameter setting was initialized.

```python
def __init__(self, data):
        # Hyper params
        self.epochs = 1000
        self.lr = 0.1
        # Network layers
        self.hidden_layer = Linear(size=100, dim=961) 
        self.logit_layer = Linear(size=1, dim=101)
        self.activation1 = Sigmoid()
        self.activation2 = Sigmoid()
```

Function `train` and `test` are in charge of the training and inference process, respectively. 

```python
def train(self):
        # Load training data
        train_data = self.toTensor(self.train_data)
        # Training epochs
        for epoch in range(self.epochs):
            # Forward
            output = self.forward(train_data)
            # Loss
            loss = self.loss_l2(output)
            # Backward
            self.backward(output)
            # Update weights
            self.optimize()
```

```python
def test(self):
        # Inference
        output = self.forward(self.toTensor(self.test_data))
        # Map to Predict Labels (Boolean)
        pred = output > 0.5
        # Map to Predict Labels (Binary)
        mapping = np.vectorize(bool2binary)
        pred_labels = mapping(pred)
        # Result
        print("Prediction:")
        print(pred_labels.reshape(-1))
        accuracy = np.sum(pred == self.test_labels) / self.num_test
        print("Accuracy: ", accuracy)
```



