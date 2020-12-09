import numpy as np
import cv2

def load_data():
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    with open("downgesture_test.list.txt") as f:
        lines = f.readlines()
        for line in lines:
            test_paths.append(line[:-1])
            test_labels.append(True if 'down' in line else False) 
            
    with open("downgesture_train.list.txt") as f:
        lines = f.readlines()
        for line in lines:
            train_paths.append(line[:-1])
            train_labels.append(1 if 'down' in line else 0) 

    return {"test_data": test_paths, 
            "test_labels": test_labels, 
            "train_data": train_paths, 
            "train_labels": train_labels}

def bool2binary(x):
    return 1 if x else 0

class Sigmoid:
    def __init__(self):
        self.out = 0

    def forward(self, x):
        # Prevent overflow.
        self.out = np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))
        return self.out
    
    def backward(self, dprev):
        return self.out * (1 - self.out) * dprev

class Linear:
    def __init__(self, size, dim):
        self.weights = 0.01 * np.random.randn(dim, size)
        self.x = 0
        self.gradient = 0
        self.dim = dim
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights)
    
    def backward(self, dprev):
        self.gradient = np.dot(self.x.T, dprev) / self.weights.shape[0]
        return np.dot(dprev, self.weights.T)[:, :-1] / dprev.shape[0]
    
    def update(self, lr):
        self.weights -= lr * self.gradient
    

class NeuralNetwork:
    '''
    Architecture: input -> hidden_layer (100) -> output
    Learning rate: 0.1
    Training epochs: 1000
    '''
    def __init__(self, data):
        # Hyper params
        self.epochs = 1000
        self.lr = 0.1

        # Data
        self.train_data = data["train_data"]
        self.test_data = data["test_data"]
        self.num_train = len(self.train_data)
        self.num_test = len(self.test_data)
        self.train_labels = np.array(data["train_labels"]).reshape(self.num_train, 1)
        self.test_labels = np.array(data["test_labels"]).reshape(self.num_test, 1)
        
        # Network layers
        self.hidden_layer = Linear(size=100, dim=961) 
        self.logit_layer = Linear(size=1, dim=101)
        self.activation1 = Sigmoid()
        self.activation2 = Sigmoid()

    def train(self):
        # Load training data
        train_data = self.toTensor(self.train_data) # 184 x 960
        # Training epochs
        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            # Forward
            output = self.forward(train_data)
            # Loss
            loss = self.loss_l2(output)
            print("loss: ", loss)
            # Backward
            self.backward(output)
            # Update weights
            self.optimize()
    
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

    def optimize(self):
        self.hidden_layer.update(self.lr)
        self.logit_layer.update(self.lr)
                
    def forward(self, input):
        N = input.shape[0]
        x = self.hidden_layer.forward(input)
        x = self.activation1.forward(x)
        x = np.concatenate((x, np.ones((N,1))), axis=1)
        x = self.logit_layer.forward(x)
        output = self.activation2.forward(x)
        return output

    def backward(self, output):
        dout = np.multiply(output - self.train_labels, 2)
        dx = self.activation2.backward(dout)
        dx = self.logit_layer.backward(dx)
        dx = self.activation1.backward(dx)
        dx = self.hidden_layer.backward(dx)

    def toTensor(self, paths):
        N = len(paths)
        tensor = np.zeros((N, 960))
        for i, path in enumerate(paths):
            img = cv2.imread(path, 0)
            img = img
            tensor[i] = img.reshape(-1)
        tensor = np.concatenate((tensor, np.ones((N,1))), axis=1)
        return tensor
    
    def loss_l2(self, output):
        return np.sum(np.power(output - self.train_labels, 2))


if __name__ == "__main__":
    data = load_data()
    nn = NeuralNetwork(data)
    nn.train()
    nn.test()
    pass