import numpy as np
# import matplotlib.pyplot as plt

class FastMap:
    def __init__(self, info, k):
        self.wordlist = info["wordlist"]
        self.N = len(self.wordlist)
        self.dis = info["distance"]
        self.k = k
        self.coord = np.zeros((self.N, self.k))

    def run(self):
        self.fastmap(self.k, self.dis)
        print('--- The coordinates matrix of words ---')
        print(self.coord)
        # self.visualize()

    def fastmap(self, k, dis):
        # Termination
        if k <= 0:
            return
        
        # The column of coordinate matrix to be updated
        col = self.k - k

        # New distance matrix
        new_dis = np.zeros_like(dis)

        # Find the farthest pair of objects
        a, b = self.choose_distant_objects(dis)
        if dis[a, b] == 0:
            self.coord[:, col] = 0
            return

        # Update coordinates of a and b
        self.coord[a, col] = 0
        self.coord[b, col] = dis[a, b]

        # Update coordinates of other objects
        for i in range(self.N):
            if i != a and i != b:
                self.coord[i, col] = self.compute_xi(dis[a, b], dis[a, i], dis[b, i])

        # Update distance matrix
        for i in range(self.N):
            for j in range(i + 1, self.N):
                new_dis[i, j] = np.sqrt(dis[i, j] ** 2 - (self.coord[i, col] - self.coord[j, col]) ** 2)
                new_dis[j, i] = new_dis[i, j]
        
        # Recursion
        self.fastmap(k - 1, new_dis)

    def choose_distant_objects(self, dis):
        return self.farthest_pair(dis, 0)
    
    def compute_xi(self, d_ab, d_ai, d_bi):
        return ((d_ai ** 2) + (d_ab ** 2) - (d_bi ** 2)) / (d_ab * 2)

    def farthest_pair(self, dis, b):
        a = np.argmax(dis, axis=0)[b]
        b_ = np.argmax(dis, axis=0)[a]
        if b == b_:
            return a, b
        return self.farthest_pair(dis, a)
    
    # def visualize(self):
    #     fig, ax = plt.subplots()
    #     xs = self.coord[:, 0]
    #     ys = self.coord[:, 1]
    #     ax.scatter(xs, ys)
    #     for i, txt in enumerate(self.wordlist):
    #         ax.annotate(txt, (xs[i], ys[i]))
    #     plt.show()


def read_file(paths):
    wordlist = []
    N = 0
    with open(paths["wordlist"]) as f:
        lines = f.readlines()
        N = len(lines)
        for line in lines:
            wordlist.append(line[: -1])
    dis = np.zeros((N, N))
    with open(paths["data"]) as f:
        lines = f.readlines()
        for line in lines:
            elems = [int(elem) for elem in line[: -1].split('\t')]
            dis[elems[0] - 1, elems[1] - 1] = elems[2]
            dis[elems[1] - 1, elems[0] - 1] = elems[2]
    return {"wordlist": wordlist, "distance": dis}

if __name__ == "__main__":
    paths = {"data": 'fastmap-data.txt', "wordlist": 'fastmap-wordlist.txt'}
    info = read_file(paths)
    fastmap = FastMap(info, 2)
    fastmap.run()
    pass