
from kmeans import KMeans
from gmm import GMM
from visualize import plot_original

def read_file(file_path):
    points = []
    xs = []
    ys = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
            points.append((float(x), float(y)))
    
    # Visualization
    # plot_original(xs, ys)

    # Scope Calculate
    x_max = max(xs)
    x_min = min(xs)
    y_max = max(ys)
    y_min = min(ys)

    return {"points": points, "x_scope":(x_min, x_max), "y_scope": (y_min, y_max)}


if __name__ == '__main__':
    # Input data from file
    info = read_file("clusters.txt")

    # K-Means
    # kmeans = KMeans(info)
    # kmeans.train()

    # GMM
    gmm = GMM(info)
    gmm.train()
