from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix


class KMeansGeodesic:

    def __init__(
        self,
        knn_graph: csr_matrix,
        distance_matrix: np.ndarray,
        n_clusters: int,
        random_state: Optional[int] = None,
        max_iters: int = 300,
        tol: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.knn_graph = knn_graph
        self.distance_matrix = distance_matrix
        self.random_state = random_state
        self.max_iters = max_iters
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None

        # Imposta il seed se fornito
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, sample_indices):
        """
        sample_indices: array degli indici dei nodi (non le coordinate!)
        """

        # Inizializza centroidi casualmente
        self.centroids = np.random.choice(sample_indices, self.n_clusters, replace=False)

        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()

            # Assign step: assegna ogni punto al centroide più vicino
            self.labels_ = self._assign_clusters(sample_indices)

            # Update step: aggiorna i centroidi
            self._update_centroids(sample_indices)

            # Check convergenza
            if np.array_equal(old_centroids, self.centroids):
                print(f"Convergenza raggiunta dopo {iteration+1} iterazioni")
                break

        # Imposta cluster_centers_ come gli indici dei centroidi
        self.cluster_centers_ = self.centroids.copy()

        return self

    def _assign_clusters(self, sample_indices):
        """Assegna ogni punto al cluster più vicino usando distanze geodesiche"""
        labels = np.zeros(len(sample_indices), dtype=int)

        for i, point_idx in enumerate(sample_indices):
            distances_to_centroids = []

            for centroid_idx in self.centroids:
                dist = self.distance_matrix[point_idx, centroid_idx]
                distances_to_centroids.append(dist)

            labels[i] = np.argmin(distances_to_centroids)

        return labels

    def _update_centroids(self, sample_indices):
        """Aggiorna i centroidi come medoidi (punto che minimizza distanza totale)"""
        new_centroids = []

        for cluster_id in range(self.n_clusters):
            # Trova tutti i punti in questo cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_points = sample_indices[cluster_mask]

            if len(cluster_points) == 0:
                # Se cluster vuoto, mantieni il centroide precedente
                new_centroids.append(self.centroids[cluster_id])
                continue

            # Trova il medoide: punto che minimizza la somma delle distanze agli altri
            min_total_distance = np.inf
            best_medoid = cluster_points[0]

            for candidate in cluster_points:
                total_distance = np.sum(
                    [self.distance_matrix[candidate, p] for p in cluster_points]
                )

                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_medoid = candidate

            new_centroids.append(best_medoid)

        self.centroids = np.array(new_centroids)
