import numpy as np
# import matplotlib.pyplot as plt
def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        points = np.zeros((len(lines), 4))
        labels = np.zeros((len(lines), 1))
        for i, line in enumerate(lines):
            elements = line.split(',')
            labels[i] = True if elements[4] == '+1\n' else False
            # Convert coordinates from str to float
            coord = [float(elem) for elem in elements[:3]]
            coord.append(1.0)
            points[i] = coord
    return {"labels": labels, "points": points}

class Pocket:
    def __init__(self, info):
        self.points = info["points"]
        self.labels = info["labels"]
        self.weights = np.random.rand(4, 1)
        self.max_iters = 7000
        self.lr = 1e-4
        self.N = self.points.shape[0]

    def train(self):
        prev_weights = self.weights
        min_error = self.error(self.weights)
        num_errors = []
        for iter in range(self.max_iters):
            weights_i = np.copy(prev_weights)
            scores = self.points.dot(prev_weights)
            scores = scores > 0
            for i in range(self.N):
                if scores[i] != self.labels[i]:
                    if self.labels[i]:
                        weights_i += (self.lr * self.points[i]).reshape(4, 1)
                    else:
                        weights_i -= (self.lr * self.points[i]).reshape((4, 1))
            prev_weights = weights_i
            error = self.error(weights_i)
            if error < min_error:
                min_error = error
                self.weights = weights_i
            num_errors.append(min_error)

        # Plot the number of misclassified points 
        # plt.scatter(np.arange(0, self.max_iters), num_errors)
        # plt.xlabel("Iterations")
        # plt.ylabel("Number of misclassified points")
        # plt.show()

        print("Weights: ")
        print(self.weights)
        print("Accuracy: ", self.get_accuracy())


    def error(self, weights):
        scores = self.points.dot(weights)
        scores = scores > 0
        return np.sum(self.labels != scores) 

    def get_accuracy(self):
        scores = self.points.dot(self.weights)
        scores = scores > 0
        return np.sum(self.labels == scores) / self.N

if __name__ == "__main__":
    info = read_file('classification.txt')
    pocket = Pocket(info)
    pocket.train()