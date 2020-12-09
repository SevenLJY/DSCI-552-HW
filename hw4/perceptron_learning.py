import numpy as np
def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        points = np.zeros((len(lines), 4))
        labels = np.zeros((len(lines), 1))
        for i, line in enumerate(lines):
            elements = line.split(',')
            # True for +1, False for -1
            labels[i] = True if elements[3] == '+1' else False
            # Convert coordinates from str to float
            coord = [float(elem) for elem in elements[:3]]
            coord.append(1.0)
            points[i] = coord
    return {"labels": labels, "points": points}

class Perceptron:
    def __init__(self, info):
        self.weights = np.random.rand(4, 1)
        self.labels = info["labels"]
        self.points = info["points"] 
        self.lr = 1e-4
        self.N = self.points.shape[0]
        self.max_iters = 10000

    def train(self):
        iter = 0
        while(self.violate_exist() and iter < self.max_iters):
            iter += 1
            scores = self.points.dot(self.weights)
            scores = scores > 0
            for i in range(self.N):
                if scores[i] != self.labels[i]:
                    if self.labels[i]:
                        self.weights += (self.lr * self.points[i]).reshape(4, 1)
                    else:
                        self.weights -= (self.lr * self.points[i]).reshape((4, 1))
        scores = self.points.dot(self.weights)
        scores = scores > 0
        print('Weight: ')
        print(self.weights)
        print('Iteration: ', iter)
        print('Accuracy', np.sum(self.labels == scores) / self.N)
        pass

    def violate_exist(self):
        scores = self.points.dot(self.weights)
        scores = scores > 0
        if np.array_equal(scores, self.labels):
            return False
        return True

if __name__ == "__main__":
    info = read_file('classification.txt')
    perceptron = Perceptron(info)
    perceptron.train()