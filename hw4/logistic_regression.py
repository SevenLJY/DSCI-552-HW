import numpy as np
def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        points = np.zeros((len(lines), 4))
        labels = np.zeros((len(lines), 1))
        for i, line in enumerate(lines):
            elements = line.split(',')
            labels[i] = 1 if elements[4] == '+1\n' else -1
            # Convert coordinates from str to float
            coord = [float(elem) for elem in elements[:3]]
            coord.append(1.0)
            points[i] = coord
    return {"labels": labels, "points": points}

class LogisticReg:
    def __init__(self, info):
        self.labels = info["labels"]
        self.points = info["points"]
        self.weights = np.random.rand(4, 1)
        self.max_iters = 7000
        self.lr = 1e-4
        self.N = self.points.shape[0]
    
    def train(self):
        for iter in range(self.max_iters):
            gradient = ( np.sum((self.labels * self.points) / (1 + np.exp(self.labels * self.points.dot(self.weights))), axis=0)/ -self.N ).reshape((4, 1))
            self.weights -= self.lr * gradient
        print('Weight: ')
        print(self.weights)
        print('Accuracy: ', self.get_accuracy())

    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))
    
    def get_accuracy(self):
        scores = self.sigmoid(self.points.dot(self.weights))
        scores = scores > 0.5
        y = self.labels == 1
        return np.sum(y == scores) / self.N

if __name__ == "__main__":
    info = read_file('classification.txt')
    logisticReg = LogisticReg(info)
    logisticReg.train()
