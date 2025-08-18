import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import warnings

class KNNGraph:
    """
    Classe per il calcolo di distanze geodesiche su grafi k-NN e clustering K-Means.
    """
    
    def __init__(self, k: int = 10):
        """
        Inizializza il grafo k-NN.
        
        Args:
            k (int): Numero di vicini più prossimi per la costruzione del grafo
        """
        self.k = k
        self.knn_model = None
        self.graph = None
        self.geodesic_distances = None
        self.data_points = None
        
    def build_knn_graph(self, data: np.ndarray, metric: str = 'euclidean') -> None:
        """
        Costruisce il grafo k-NN dai dati.
        
        Args:
            data (np.ndarray): Dati di input (n_samples, n_features)
            metric (str): Metrica per il calcolo delle distanze ('euclidean', 'cosine', etc.)
        """
        print(f"Costruisco il grafo k-NN con k={self.k} per {data.shape[0]} punti...")
        
        self.data_points = data
        
        # Costruisce il modello k-NN
        self.knn_model = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 perché include il punto stesso
            metric=metric,
            n_jobs=-1
        )
        self.knn_model.fit(data)
        
        # Trova i k vicini più prossimi per ogni punto
        distances, indices = self.knn_model.kneighbors(data)
        
        # Rimuove il primo vicino (il punto stesso)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Costruisce la matrice di adiacenza sparsa
        n_samples = data.shape[0]
        row_indices = []
        col_indices = []
        edge_weights = []
        
        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                row_indices.append(i)
                col_indices.append(neighbor_idx)
                edge_weights.append(distances[i, j])
                
                # Rende il grafo non diretto aggiungendo l'arco opposto
                row_indices.append(neighbor_idx)
                col_indices.append(i)
                edge_weights.append(distances[i, j])
        
        # Crea la matrice sparsa
        self.graph = csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        
        # Rimuove duplicati e mantiene il peso minimo per archi duplicati
        self.graph.eliminate_zeros()
        
        print(f"Grafo k-NN costruito: {self.graph.nnz} archi")
        
    def compute_geodesic_distances(self) -> np.ndarray:
        """
        Calcola le distanze geodesiche (shortest path) sul grafo k-NN.
            
        Returns:
            np.ndarray: Matrice delle distanze geodesiche (n_samples, n_samples)
        """
        if self.graph is None:
            raise ValueError("Il grafo deve essere costruito prima di calcolare le distanze geodesiche")

        print(f"Calcolo delle distanze geodesiche con algoritmo Dijkstra...")

        # Calcola le distanze shortest path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.geodesic_distances = shortest_path(
                csgraph=self.graph,
                method='D',
                directed=False
            )
        
        # Gestisce i componenti disconnessi
        infinite_mask = np.isinf(self.geodesic_distances)
        if np.any(infinite_mask):
            n_infinite = np.sum(infinite_mask)
            print(f"Attenzione: {n_infinite} coppie di punti non sono connesse nel grafo")
            
            # Sostituisce gli infiniti con una distanza molto grande
            max_finite_dist = np.max(self.geodesic_distances[~infinite_mask])
            self.geodesic_distances[infinite_mask] = max_finite_dist * 10
        
        print(f"Distanze geodesiche calcolate. Range: [{np.min(self.geodesic_distances):.3f}, {np.max(self.geodesic_distances):.3f}]")
        
        return self.geodesic_distances
    
    def fit(self, data: np.ndarray, metric: str = 'euclidean') -> None:
        self.build_knn_graph(data, metric)
        self.compute_geodesic_distances()
        
        return self
    