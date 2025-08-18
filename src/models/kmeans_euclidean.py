from typing import Optional
from sklearn.cluster import KMeans


class KMeansEuclidean:

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        max_iters: int = 300,
        tol: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, samples):
        """
        X_indices: array degli indici dei nodi (non le coordinate!)
        """

        self.kmeans_alg = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.euclidean_labels = self.kmeans_alg.fit_predict(samples)

        return self
