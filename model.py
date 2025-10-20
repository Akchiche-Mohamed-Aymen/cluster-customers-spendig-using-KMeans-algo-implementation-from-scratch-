import numpy as np
import matplotlib.pyplot as plt
from random import randint
def _standardize(X , mean , std):
        return (X - mean) / std
    
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def KMeans_plus_plus(X , k):
    idx  = randint(0, len(X) - 1)
    cl = [X[idx]]
    for _ in range(1,k):
        distances =[min([euclidean_distance(i , c) for c in cl])**2 for i in X]
        p = sum(distances)
        distances = np.array([k / p for k in distances])
        i = np.argmax(distances)
        cl.append(X[i])
    return cl

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean vector) for each cluster
        self.centroids = []

    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X = _standardize(X , self.mean , self.std)
        self.n_samples, self.n_features = X.shape
        # initialize
        self.centroids = KMeans_plus_plus(self.X , self.K)
        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        self.labels  =  self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx


    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) < 0.1

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

X = np.array([
    [1,  2,  3,  4,  5,  6],
    [2,  3,  4,  5,  6,  7],
    [3,  4,  5,  6,  7,  8],
    [10, 11, 12, 13, 14, 15],
    [11, 12, 13, 14, 15, 16],
    [12, 13, 14, 15, 16, 17],
    [13, 14, 15, 16, 17, 18],
    [20, 21, 22, 23, 24, 25],
    [21, 22, 23, 24, 25, 26],
    [22, 23, 24, 25, 26, 27],
])
k = 3
KMeans_plus_plus(X , k)