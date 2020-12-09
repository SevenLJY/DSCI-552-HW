import numpy as np

def read_file(path):
    points = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            coords = [float(coord) for coord in line[: -1].split('\t')]
            points.append(coords)
    return np.array(points)

class PCA:
    def __init__(self, data):
        self.points, self.N = self.preprocess(data)
    
    def preprocess(self, x):
        N, _ = x.shape
        mean = x.sum(axis=0) / N
        return x - mean, N
    
    def run(self):
        cov = self.points.T.dot(self.points) / self.N
        _, eigenvectors = np.linalg.eig(cov)
        print('The First Two principle Components')
        print(eigenvectors[:, 0: 2])


if __name__ == "__main__":
    data = read_file('pca-data.txt')
    pca = PCA(data)
    pca.run()
    pass