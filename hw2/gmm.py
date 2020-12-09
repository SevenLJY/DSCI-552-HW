import random
import numpy as np
from visualize import plot_clusters

def Gaussian_2D(mean, cov, amplitude, point):
    A = amplitude / np.sqrt( 2 * np.pi * np.linalg.det(cov)) 
    return A * np.exp(-0.5 * np.dot(point - mean, np.linalg.inv(cov)).dot((point-mean).T))

class GMM:
    def __init__(self, info):
        self.K = 3
        self.threshold = 1e-7                                     # threshold for convergence
        self.points = np.array(info["points"])                    # numpy array of data points (N x D) 
        self.N, self.dim = self.points.shape                      # markdown the shape of data array
        self.weights = self.random_guess()                        # numpy array of weights (N x K)
        self.means = np.zeros((self.K, self.dim))                 # numpy array of means (K x D)
        self.covariances = np.zeros((self.K, self.dim, self.dim)) # numpy array of covariance matrix (K x D x D)        
        self.amplitudes = np.zeros((self.K,))                     # numpy array of amplitudes (K, )

    def random_guess(self):
        weights = []
        for i in range(self.points.shape[0]):
            probs = random.sample(range(0, 10), self.K)
            normalize_factor = sum(probs)
            probs = [prob / normalize_factor for prob in probs]
            weights.append(probs)
        return np.array(weights)
            
    def train(self):
        self.clustering()

        # print out the gaussians
        print('----------- mean -----------')
        print(self.means)
        print('-------- covariance --------')
        print(self.covariances)
        print('--------- amplitude --------')
        print(self.amplitudes)

        # # Visualization
        # assignments = self.assign_cluster()
        # plot_clusters(self.points, assignments)

    def clustering(self, stop_flag=False):
        # Check Termination
        if stop_flag:
            return
        
        # M Step
        means, covariances, amplitudes = self.m_step()

        # E Step
        weights = self.e_step(means, covariances, amplitudes)

        # Check convergence
        need_stop = self.check_convergence(weights)

        # Update variables
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.amplitudes = amplitudes
        
        # Recursion
        self.clustering(need_stop)

    
    def m_step(self):
        # Weighted mean
        means = self.weighted_mean()

        # Amplitudes
        amplitudes = np.sum(self.weights, axis=0) / self.N

        # Weighted covariance
        covariances = self.weighted_covariance(means)

        return means, covariances, amplitudes

    def weighted_mean(self):
        means = np.zeros_like(self.means)
        for c in range(self.K):
            for i in range(self.N):
                means[c] += self.points[i] * self.weights[i, c]
        sum_weights = np.sum(self.weights, axis=0)
        return (means.T / sum_weights).T

    def weighted_covariance(self, means):
        sum_weights = np.sum(self.weights, axis=0)
        covariances = np.zeros_like(self.covariances)
        for c in range(self.K):
            for i in range(self.N):
                D = (self.points[i] - means[c]).reshape((self.dim, 1))
                covariances[c] += np.dot(D, D.T) * self.weights[i, c]
            covariances[c] /= sum_weights[c]
        return covariances

    def e_step(self, means, covariances, amplitudes):
        weights = np.zeros_like(self.weights)
        for c in range(self.K):
            mean = means[c]
            cov = covariances[c]
            amplitude = amplitudes[c]
            for i, point in enumerate(self.points):
                weights[i, c] = Gaussian_2D(mean, cov, amplitude, point)
        normalize_factor = np.sum(weights, axis=1)
        return (weights.T / normalize_factor).T
    
    def check_convergence(self, weights):
        return np.all(np.abs((weights - self.weights)) - self.threshold <= 0)

    def assign_cluster(self):
        assignments = [[],[],[]]
        point2idx = np.argmax(self.weights, axis=1)
        for i in range(self.N):
            assignments[point2idx[i]].append(i)
        return assignments