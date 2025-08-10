import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

class GeodesicKNNGraph:
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
        
    def compute_geodesic_distances(self, method: str = 'D') -> np.ndarray:
        """
        Calcola le distanze geodesiche (shortest path) sul grafo k-NN.
        
        Args:
            method (str): Algoritmo per il calcolo dello shortest path ('D' per Dijkstra, 'FW' per Floyd-Warshall)
            
        Returns:
            np.ndarray: Matrice delle distanze geodesiche (n_samples, n_samples)
        """
        if self.graph is None:
            raise ValueError("Il grafo deve essere costruito prima di calcolare le distanze geodesiche")
        
        print(f"Calcolo delle distanze geodesiche con algoritmo {method}...")
        
        # Calcola le distanze shortest path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.geodesic_distances = shortest_path(
                csgraph=self.graph,
                method=method,
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
    
    def geodesic_kmeans(self, n_clusters: int, random_state: Optional[int] = None, 
                       max_iter: int = 300, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Esegue K-Means usando distanze geodesiche.
        
        Args:
            n_clusters (int): Numero di cluster
            random_state (int, optional): Seed per la riproducibilità
            max_iter (int): Numero massimo di iterazioni
            tol (float): Tolleranza per la convergenza
            
        Returns:
            tuple: (cluster_labels, cluster_centers_indices)
        """
        if self.geodesic_distances is None:
            raise ValueError("Le distanze geodesiche devono essere calcolate prima del clustering")
        
        print(f"Eseguo K-Means geodesico con {n_clusters} cluster...")
        
        n_samples = self.geodesic_distances.shape[0]
        
        # Inizializzazione casuale dei centri
        if random_state is not None:
            np.random.seed(random_state)
        
        center_indices = np.random.choice(n_samples, n_clusters, replace=False)
        labels = np.zeros(n_samples, dtype=int)
        
        for iteration in range(max_iter):
            old_labels = labels.copy()
            
            # Assegnazione: ogni punto al centro più vicino (distanza geodesica)
            for i in range(n_samples):
                distances_to_centers = [self.geodesic_distances[i, center] for center in center_indices]
                labels[i] = np.argmin(distances_to_centers)
            
            # Aggiornamento centri: punto con distanza geodesica media minima nel cluster
            new_center_indices = []
            for cluster_id in range(n_clusters):
                cluster_points = np.where(labels == cluster_id)[0]
                
                if len(cluster_points) == 0:
                    # Se il cluster è vuoto, sceglie un punto casuale
                    new_center_indices.append(np.random.choice(n_samples))
                    continue
                
                # Trova il punto con distanza media minima dagli altri punti del cluster
                min_avg_dist = np.inf
                best_center = cluster_points[0]
                
                for candidate in cluster_points:
                    avg_dist = np.mean([self.geodesic_distances[candidate, p] for p in cluster_points])
                    if avg_dist < min_avg_dist:
                        min_avg_dist = avg_dist
                        best_center = candidate
                
                new_center_indices.append(best_center)
            
            center_indices = np.array(new_center_indices)
            
            # Controllo convergenza
            if np.array_equal(labels, old_labels):
                print(f"Convergenza raggiunta dopo {iteration + 1} iterazioni")
                break
            
            # Controllo tolleranza
            changes = np.sum(labels != old_labels)
            if changes / n_samples < tol:
                print(f"Convergenza raggiunta (tolleranza) dopo {iteration + 1} iterazioni")
                break
        
        print(f"K-Means completato. Distribuzione cluster: {np.bincount(labels)}")
        
        return labels, center_indices
    
    def compare_with_euclidean_kmeans(self, n_clusters: int, random_state: Optional[int] = None) -> dict:
        """
        Confronta il clustering geodesico con il K-Means euclideo standard.
        
        Args:
            n_clusters (int): Numero di cluster
            random_state (int, optional): Seed per la riproducibilità
            
        Returns:
            dict: Risultati del confronto
        """
        if self.data_points is None:
            raise ValueError("I dati devono essere disponibili per il confronto")
        
        print("Confronto con K-Means euclideo...")
        
        # K-Means euclideo standard
        kmeans_euclidean = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        euclidean_labels = kmeans_euclidean.fit_predict(self.data_points)
        
        # K-Means geodesico
        geodesic_labels, geodesic_centers = self.geodesic_kmeans(n_clusters, random_state)
        
        # Calcola metriche di confronto
        results = {
            'euclidean_labels': euclidean_labels,
            'geodesic_labels': geodesic_labels,
            'geodesic_centers': geodesic_centers,
            'euclidean_centers': kmeans_euclidean.cluster_centers_,
            'euclidean_inertia': kmeans_euclidean.inertia_,
            'n_different_assignments': np.sum(euclidean_labels != geodesic_labels),
            'agreement_percentage': np.mean(euclidean_labels == geodesic_labels) * 100
        }
        
        print(f"Accordo tra clustering euclideo e geodesico: {results['agreement_percentage']:.1f}%")
        print(f"Punti assegnati diversamente: {results['n_different_assignments']}")
        
        return results
    
    def visualize_graph_connectivity(self, sample_size: int = 1000) -> None:
        """
        Visualizza la connettività del grafo k-NN.
        
        Args:
            sample_size (int): Numero di punti da campionare per la visualizzazione
        """
        if self.graph is None:
            raise ValueError("Il grafo deve essere costruito prima della visualizzazione")
        
        n_samples = self.graph.shape[0]
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            subgraph = self.graph[np.ix_(indices, indices)]
        else:
            subgraph = self.graph
            indices = np.arange(n_samples)
        
        # Analizza la connettività
        n_components, component_labels = connected_components(subgraph, directed=False)
        
        print(f"Analisi connettività del grafo (campione di {len(indices)} punti):")
        print(f"Numero di componenti connesse: {n_components}")
        
        if n_components > 1:
            component_sizes = np.bincount(component_labels)
            print(f"Dimensioni componenti: {sorted(component_sizes, reverse=True)}")
        
        # Visualizza la distribuzione dei gradi
        degrees = np.array(subgraph.sum(axis=1)).flatten()
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(degrees, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Grado del nodo')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione dei gradi nel grafo k-NN')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(n_components), sorted(np.bincount(component_labels), reverse=True))
        plt.xlabel('Componente connessa')
        plt.ylabel('Numero di nodi')
        plt.title('Dimensioni delle componenti connesse')
        
        plt.tight_layout()
        plt.show()

def connected_components(csgraph, directed=True):
    """
    Wrapper per scipy.sparse.csgraph.connected_components per compatibilità.
    """
    return connected_components(csgraph, directed=directed)
