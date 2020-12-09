from visualize import plot_original, plot_centroids, plot_clusters
import random
class KMeans:
    def __init__(self, info):
        self.K = 3
        self.points = info["points"]                                          # data points
        self.centroids = self.init_centroid(info["x_scope"], info["y_scope"]) # K centroids
        self.assignments = []                                                 # Just for visualization

    def init_centroid(self, x_scope, y_scope):
        centroids = []
        for i in range(self.K):
            x = random.uniform(x_scope[0], x_scope[1])
            y = random.uniform(y_scope[0], y_scope[1])
            centroids.append((x, y))
        return centroids
    
    def train(self):
        self.clustering()
        print("------Final Centroids-----")
        for centroid in self.centroids:
            print(centroid)
        
        # Visualization
        plot_centroids(self.points, self.centroids)
        plot_clusters(self.points, self.assignments)
        
        pass

    def clustering(self, need_stop=False):
        # Termination check
        if need_stop:
            return
        
        # Init assignment list: points mapping to the closest centroid
        assignments = []
        for i in range(self.K):
            assignments.append([])

        # Assign data points to the closest centroid
        for idx, point in enumerate(self.points):
            closest = self.find_closest(point)
            assignments[closest].append(idx)
        
        # Just for clustering visualization
        self.assignments = assignments
        
        # Recalculate the centroids
        new_centroids = []
        new_centroids = self.calculate_centroid(assignments)

        # Check if reach the stable status
        # If not, update the centroids
        # Otherwise, update the stop flag
        stop_flag = self.check_stable(new_centroids) 

        if not stop_flag:
            self.centroids = new_centroids
        
        # Recursion
        self.clustering(stop_flag)

        return

    def find_closest(self, point):
        closest = 0
        min_dis = float('inf') 
        for idx, centroid in enumerate(self.centroids):
            dis = (point[0] - centroid[0])**2 + (point[1] - centroid[1])**2
            if dis < min_dis:
                closest = idx
                min_dis = dis
        return closest


    def calculate_centroid(self, assignments):
        centroids = []
        for i in range(self.K):
            centroid = self.mean(assignments[i])
            centroids.append(centroid)
        return centroids
    
    def mean(self, indices):
        if len(indices) == 0:
            return self.points[0]
        sum_x = 0
        sum_y = 0
        for idx in indices:
            sum_x += self.points[idx][0]
            sum_y += self.points[idx][1]
        return (sum_x / len(indices), sum_y / len(indices))

    def check_stable(self, new_centroids):
        for idx, centroid in enumerate(new_centroids):
            if centroid[0] != self.centroids[idx][0] or centroid[1] != self.centroids[idx][1]:
                return False
        return True
