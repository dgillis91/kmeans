"""
NOTES:
    The epochs are implemented in the simple KMeans. With this strategy,
    we naively overcome the problems of poor initialization. Worth noting
    that this slows the algorithm down. For large datasets, this is obviously
    problematic. Furthermore, the initialization isn't enhanced between epochs.
"""
import numpy as np
from scipy.spatial.distance import cityblock, euclidean


metrics = {
    'manhattan': cityblock,
    'euclidean': euclidean
}


def make_2d_array(data: np.ndarray) -> np.ndarray:
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data, -1)
    return data


def sum_squared_error(points: np.ndarray) -> float:
    points = make_2d_array(points)
    centroid = np.mean(points, 0)
    errors = np.linalg.norm(points-centroid, ord=2, axis=1)
    return np.sum(errors)


class KMeans:
    def __init__(self, k=2):
        self.k = k

    def random_centroids(self, points: np.ndarray) -> np.ndarray:
        np.random.shuffle(points)
        return points[0:self.k]

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class SimpleKMeans(KMeans):
    def __init__(self, k=2, epochs=10, max_iters=100, keep_training_history=True, distance_metric='euclidean'):
        self.k = k
        self.epochs = epochs
        self.max_iters = max_iters
        self._clusters = [None]
        self.distance_metric = metrics.get(distance_metric)
        if self.distance_metric is None:
            raise ValueError('Invalid distance metric')
        if keep_training_history:
            self._training_epoch_history = []
            self._training_iteration_history = []
            self._training_sse_history = []
        else:
            self._training_epoch_history = None

    @property
    def training_history(self):
        return self._training_epoch_history, self._training_iteration_history, self._training_sse_history

    @property
    def clusters(self):
        return self._clusters

    def fit(self, predictors):
        """
        :param predictors:
        :return:
        """
        points = make_2d_array(np.copy(predictors))
        assert len(predictors) >= self.k

        best_sse = np.inf
        for epoch in range(self.epochs):
            centroids = self.random_centroids(points)

            last_sse = np.inf
            for iteration in range(self.max_iters):
                clusters = [None] * self.k
                for point in points:
                    index = np.argmin([
                        self.distance_metric(centroid, point) for centroid in centroids
                    ])
                    if clusters[index] is None:
                        clusters[index] = np.expand_dims(point, 0)
                    else:
                        clusters[index] = np.vstack((clusters[index], point))

                centroids = [np.mean(cluster, 0) for cluster in clusters]

                sse = np.sum([sum_squared_error(cluster) for cluster in clusters])
                delta = last_sse - sse
                if sse < best_sse:
                    best_clusters, best_sse = clusters, sse

                if self._training_epoch_history is not None:
                    self._training_epoch_history.append(epoch)
                    self._training_iteration_history.append(iteration)
                    self._training_sse_history.append(sse)

                if np.isclose(delta, 0, atol=0.0001):
                    break
                last_sse = sse

        self._clusters = best_clusters

    def predict(self, predictors):
        pass


class BisectingKMeans(KMeans):
    def __init__(self, k=2, max_iters=10, distance_metric='euclidean'):
        self.k = k
        self.max_iters = max_iters
        self._clusters = [None]
        self.distance_metric = metrics.get(distance_metric)
        if self.distance_metric is None:
            raise ValueError('Invalid distance metric')

    @property
    def clusters(self):
        return self._clusters

    def fit(self, predictors):
        predictors = make_2d_array(np.copy(predictors))
        self._clusters = [predictors]

        while len(self._clusters) < self.k:
            next_cluster_index = np.argmax([sum_squared_error(cluster) for cluster in self._clusters])
            split_cluster = self._clusters.pop(next_cluster_index)
            c = SimpleKMeans(k=2, keep_training_history=False)
            c.fit(split_cluster)
            self._clusters.extend(c.clusters)

    def predict(self, predictors):
        pass


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns

    np.random.seed(42)

    n_clusters = 4

    x, y = make_blobs(
        n_samples=20, centers=n_clusters, center_box=(-1, 100),
        cluster_std=10
    )
    euclid_clusterer = BisectingKMeans(k=n_clusters)
    euclid_clusterer.fit(predictors=x)

    for cluster in euclid_clusterer.clusters:
        points = make_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()
    sns.scatterplot(x[:, 0], x[:, 1], hue=y)
    plt.show()

    manhattan_clusterer = BisectingKMeans(k=n_clusters, distance_metric='manhattan')
    manhattan_clusterer.fit(predictors=x)
    for cluster in manhattan_clusterer.clusters:
        points = make_2d_array(cluster)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros_like(points)])
        plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()