import matplotlib.pyplot as plt
def plot_original(x, y):
    plt.scatter(x, y)
    plt.show()

def plot_centroids(points, centroids):
    point_xs = []
    point_ys = []
    for point in points:
        point_xs.append(point[0])
        point_ys.append(point[1])
    
    centroid_xs = []
    centroid_ys = []
    for centroid in centroids:
        centroid_xs.append(centroid[0])
        centroid_ys.append(centroid[1])

    data = ((point_xs, point_ys), (centroid_xs, centroid_ys))
    colors = ("steelblue","firebrick")
    groups = ("data points", "centroids")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, c=color, edgecolors='none', label=group)

    plt.legend(loc=2)
    plt.show()

def plot_clusters(points, assignments):
    clusters = []
    for assignment in assignments:
        cluster_x = []
        cluster_y = []
        for idx in assignment:
            cluster_x.append(points[idx][0])
            cluster_y.append(points[idx][1])
        clusters.append((cluster_x, cluster_y))
    colors = ("steelblue","firebrick","seagreen")
    groups = ("Cluster 1", "Cluster 2", "Cluster 3")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(clusters, colors, groups):
        x, y = data
        ax.scatter(x, y, c=color, edgecolors='none', label=group)

    plt.legend(loc=2)
    plt.show()
    