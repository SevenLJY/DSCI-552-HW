import numpy as np
from cvxopt import matrix, solvers

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        N = len(lines)
        points = np.zeros((N, 2))
        labels = np.zeros((N, 1))
        for i, line in enumerate(lines):
            tokens = line.split(',')
            points[i] = [float(x) for x in tokens[:2]]
            labels[i] = 1 if tokens[-1][:-1] == '+1' else -1
    return {"points": points, "labels": labels}

class SVM_Kernel:
    def __init__(self, info):
        self.points = info["points"]
        self.labels = info["labels"]
        self.N = self.labels.shape[0]
    
    def run(self):
        # Polynomial Kernel function
        z = self.poly_kernel()
        # Invoke QPP
        sol = self.QP(z)
        alpha = np.array(sol['x'])
        # Retrieve W
        W = np.sum(alpha * z * self.labels, axis=0)
        # Retrieve b
        sv, b = self.calc_b(alpha, W, z)
        
        # Print out
        print('W:', W)
        print('b: ', b)
        print("Support Vectors: ")
        print(sv)
    
    def poly_kernel(self):
        '''
        The kernel funtion: (1 + X^T X) ^ 2
        '''
        z = np.ones((self.N, 6))
        x = self.points[:, 0]
        y = self.points[:, 1]
        z[:, 1] = x ** 2
        z[:, 2] = y ** 2
        z[:, 3] = np.sqrt(2) * x
        z[:, 4] = np.sqrt(2) * y
        z[:, 5] = np.sqrt(2) * x * y
        return z
    
    def QP(self, z):
        P = matrix(np.dot(z, z.T) * np.dot(self.labels, self.labels.T))
        q = matrix(np.ones(100) * -1)
        G = matrix(np.diag(np.ones(100) * -1))
        h = matrix(np.zeros(100))
        b = matrix([0.0])
        A = matrix(self.labels.T, (1, 100))
        return solvers.qp(P, q, G, h, A, b)
    
    def calc_b(self, alpha, W, z):
        alpha_non_zero_idx = np.argwhere(alpha > 1e-6)
        sv = self.points[alpha_non_zero_idx[:, 0]]
        zm = z[alpha_non_zero_idx[:, 0]]
        b = self.labels[alpha_non_zero_idx[0][0]] - np.dot(W, zm[0].T)
        return sv, b

if __name__ == "__main__":
    info = read_file('linsep.txt')
    svm_k = SVM_Kernel(info)
    svm_k.run()