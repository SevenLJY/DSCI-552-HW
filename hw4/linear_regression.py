import numpy as np
def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        N = len(lines)
        y = np.zeros((N, 1))
        D = np.zeros((N, 3))
        for i, line in enumerate(lines):
            elems = line.split(',')
            coord = [float(elem) for elem in elems[:2]]
            coord.append(1.0)
            D[i] = coord
            y[i] = float(elems[-1][:-1])
    return {"data": D, "y": y}
    

class LinearReg:
    def __init__(self, info):
        self.D = info["data"]
        self.y = info["y"]
        self.weights = np.zeros((3, 1))
    
    def train(self):
        self.weights = np.linalg.inv(np.dot(self.D.T,self.D)).dot(self.D.T).dot(self.y)
        print(self.weights)

if __name__ == "__main__":
    info = read_file('linear-regression.txt')
    linearReg = LinearReg(info)
    linearReg.train()
    pass