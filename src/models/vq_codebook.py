import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path


class GeodesicVectorQuantizer(nn.Module):
    """
    Vector Quantizer che utilizza un codebook derivato dal clustering geodesico.
    Il codebook contiene i centroidi ottenuti dal clustering k-NN geodesico.
    """
    
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Inizializza il Vector Quantizer.
        
        Args:
            codebook_size (int): Numero di vettori nel codebook (numero di cluster)
            embedding_dim (int): Dimensione dei vettori di embedding
            commitment_cost (float): Peso del commitment loss
        """
        super(GeodesicVectorQuantizer, self).__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Inizializza il codebook come parametro del modello
        self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim))
        
        # Per memorizzare le distanze geodesiche tra i vettori del codebook
        self.geodesic_distances = None
        self.knn_graph_data = None
        
    def load_from_clustering_results(self, clustering_file: str):
        """
        Carica il codebook dai risultati del clustering geodesico.
        
        Args:
            clustering_file (str): Path al file .npz contenente i risultati del clustering
        """
        print(f"Caricamento codebook da: {clustering_file}")
        
        # Carica i dati del clustering
        data = np.load(clustering_file, allow_pickle=True)
        
        latent_vectors = data['latent_vectors']
        geodesic_labels = data['geodesic_labels']
        geodesic_centers = data['geodesic_centers']
        
        # Estrai i centroidi reali dai vettori latenti
        unique_labels = np.unique(geodesic_labels)
        n_clusters = len(unique_labels)
        
        print(f"Numero di cluster nel file: {n_clusters}")
        print(f"Dimensione vettori latenti: {latent_vectors.shape[1]}")
        
        # Calcola i centroidi come media dei punti in ogni cluster
        centroids = []
        for label in unique_labels:
            cluster_points = latent_vectors[geodesic_labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Aggiorna i parametri del modello
        self.codebook_size = n_clusters
        self.embedding_dim = centroids.shape[1]
        
        # Aggiorna il codebook con i centroidi calcolati
        with torch.no_grad():
            self.codebook = nn.Parameter(torch.from_numpy(centroids).float())
        
        # Memorizza informazioni aggiuntive
        self.knn_graph_data = {
            'latent_vectors': latent_vectors,
            'geodesic_labels': geodesic_labels,
            'geodesic_centers': geodesic_centers,
            'config': data['config'].item() if 'config' in data else None
        }
        
        print(f"Codebook caricato con successo!")
        print(f"  • Dimensione codebook: {self.codebook_size}")
        print(f"  • Dimensione embedding: {self.embedding_dim}")
        
        # Calcola statistiche del codebook
        codebook_np = self.codebook.detach().numpy()
        print(f"  • Range valori codebook: [{np.min(codebook_np):.3f}, {np.max(codebook_np):.3f}]")
        print(f"  • Norma media vettori: {np.mean(np.linalg.norm(codebook_np, axis=1)):.3f}")
        
    def compute_geodesic_distances_between_codes(self, k_neighbors: int = 10):
        """
        Calcola le distanze geodesiche tra i vettori del codebook utilizzando
        un grafo k-NN costruito sui centroidi.
        
        Args:
            k_neighbors (int): Numero di vicini per il grafo k-NN
        """
        from models.knn_graph import GeodesicKNNGraph
        
        print(f"Calcolo distanze geodesiche tra {self.codebook_size} vettori del codebook...")
        
        codebook_np = self.codebook.detach().numpy()
        
        # Costruisci il grafo k-NN sui centroidi del codebook
        # Usa k minore se abbiamo pochi centroidi
        effective_k = min(k_neighbors, self.codebook_size - 1)
        
        graph = GeodesicKNNGraph(k=effective_k)
        graph.build_knn_graph(codebook_np, metric='euclidean')
        
        # Calcola le distanze geodesiche
        self.geodesic_distances = graph.compute_geodesic_distances(method='D')
        
        print(f"Distanze geodesiche calcolate per il codebook")
        print(f"  • Range distanze: [{np.min(self.geodesic_distances):.3f}, {np.max(self.geodesic_distances):.3f}]")
        
    def quantize(self, inputs: torch.Tensor, use_geodesic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizza i vettori di input utilizzando il codebook.
        
        Args:
            inputs (torch.Tensor): Vettori da quantizzare [batch_size, embedding_dim]
            use_geodesic (bool): Se utilizzare distanze geodesiche per la quantizzazione
            
        Returns:
            tuple: (quantized_vectors, quantization_loss, encoding_indices)
        """
        # Assicurati che input e codebook siano sullo stesso dispositivo
        device = inputs.device
        if self.codebook.device != device:
            self.codebook = self.codebook.to(device)
        
        # Flatten dell'input se necessario
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        if use_geodesic and self.geodesic_distances is not None:
            # Quantizzazione usando distanze geodesiche
            encoding_indices = self._quantize_geodesic(flat_input)
        else:
            # Quantizzazione euclidea standard
            encoding_indices = self._quantize_euclidean(flat_input)
        
        # Ottieni i vettori quantizzati
        quantized = self.codebook[encoding_indices]
        
        # Calcola il loss di quantizzazione
        quantization_loss = self._compute_quantization_loss(flat_input, quantized)
        
        # Straight-through estimator: stop gradient per il forward pass
        quantized = flat_input + (quantized - flat_input).detach()
        
        # Ripristina la forma originale
        quantized = quantized.view(input_shape)
        
        return quantized, quantization_loss, encoding_indices
    
    def _quantize_euclidean(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Quantizzazione usando distanze euclidee standard.
        """
        # Calcola le distanze euclidee tra input e codebook
        distances = torch.cdist(inputs, self.codebook)
        
        # Trova l'indice del vettore più vicino
        encoding_indices = torch.argmin(distances, dim=1)
        
        return encoding_indices
    
    def _quantize_geodesic(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Quantizzazione usando distanze geodesiche (approssimazione).
        Prima trova i k codebook più vicini in spazio euclideo,
        poi seleziona quello con distanza geodesica minima.
        """
        if self.geodesic_distances is None:
            print("Attenzione: distanze geodesiche non disponibili, uso distanze euclidee")
            return self._quantize_euclidean(inputs)
        
        device = inputs.device
        
        # Per ogni input, trova i k candidati più vicini (euclideo)
        k_candidates = min(5, self.codebook_size)  # Considera i 5 più vicini
        
        distances_euclidean = torch.cdist(inputs, self.codebook)
        _, top_k_indices = torch.topk(distances_euclidean, k_candidates, dim=1, largest=False)
        
        batch_size = inputs.shape[0]
        encoding_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Per ogni input, scegli tra i candidati usando distanze geodesiche
        for i in range(batch_size):
            candidates = top_k_indices[i]
            
            # Calcola le "distanze geodesiche approssimate" ai candidati
            # Usa la distanza euclidea pesata dalle distanze geodesiche nel codebook
            euclidean_dists = distances_euclidean[i, candidates]
            
            # Peso basato sulla centralità geodesica nel codebook
            geodesic_weights = []
            for candidate in candidates:
                # Distanza media dal candidato agli altri vettori del codebook
                avg_geodesic_dist = np.mean(self.geodesic_distances[candidate])
                geodesic_weights.append(avg_geodesic_dist)
            
            geodesic_weights = torch.tensor(geodesic_weights, device=device)
            
            # Combina distanza euclidea e peso geodesico
            combined_scores = euclidean_dists + 0.1 * geodesic_weights
            
            # Seleziona il candidato con score minimo
            best_candidate_idx = torch.argmin(combined_scores)
            encoding_indices[i] = candidates[best_candidate_idx]
        
        return encoding_indices
    
    def _compute_quantization_loss(self, inputs: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Calcola il loss di quantizzazione (Vector Quantization Loss + Commitment Loss).
        """
        # VQ Loss: muove il codebook verso gli encoder outputs
        vq_loss = torch.mean((quantized.detach() - inputs) ** 2)
        
        # Commitment Loss: incoraggia l'encoder a committare al codebook
        commitment_loss = torch.mean((quantized - inputs.detach()) ** 2)
        
        total_loss = vq_loss + self.commitment_cost * commitment_loss
        
        return total_loss
    
    def get_codebook_usage(self, encoding_indices: torch.Tensor) -> dict:
        """
        Analizza l'utilizzo del codebook dato un set di indici di encoding.
        
        Args:
            encoding_indices (torch.Tensor): Indici dei vettori del codebook utilizzati
            
        Returns:
            dict: Statistiche sull'utilizzo del codebook
        """
        if encoding_indices.numel() == 0:
            return {'usage_rate': 0.0, 'used_codes': 0, 'total_codes': self.codebook_size}
        
        unique_indices = torch.unique(encoding_indices)
        usage_rate = len(unique_indices) / self.codebook_size
        
        # Calcola la distribuzione di utilizzo
        usage_counts = torch.bincount(encoding_indices.flatten(), minlength=self.codebook_size)
        
        return {
            'usage_rate': usage_rate,
            'used_codes': len(unique_indices),
            'total_codes': self.codebook_size,
            'usage_distribution': usage_counts.cpu().numpy(),
            'most_used_code': torch.argmax(usage_counts).item(),
            'least_used_codes': (usage_counts == 0).sum().item()
        }
    
    def visualize_codebook(self, save_path: Optional[str] = None):
        """
        Visualizza il codebook usando t-SNE o PCA.
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        codebook_np = self.codebook.detach().numpy()
        
        # Se la dimensione è maggiore di 2, usa t-SNE per la visualizzazione
        if self.embedding_dim > 2:
            print("Applicando t-SNE per visualizzazione 2D del codebook...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, self.codebook_size-1))
            codebook_2d = tsne.fit_transform(codebook_np)
        else:
            codebook_2d = codebook_np
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], 
                            c=range(self.codebook_size), cmap='tab20', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Aggiungi etichette ai punti
        for i, (x, y) in enumerate(codebook_2d):
            plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.title(f'Codebook Visualization ({self.codebook_size} codes, {self.embedding_dim}D)')
        plt.xlabel('Dimension 1' if self.embedding_dim > 2 else 'Actual Dimension 1')
        plt.ylabel('Dimension 2' if self.embedding_dim > 2 else 'Actual Dimension 2')
        plt.colorbar(scatter, label='Code Index')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizzazione codebook salvata in: {save_path}")
        
        plt.show()
    
    def save_codebook(self, path: str):
        """
        Salva il codebook su disco.
        """
        torch.save({
            'codebook': self.codebook.detach(),
            'codebook_size': self.codebook_size,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost,
            'geodesic_distances': self.geodesic_distances,
            'knn_graph_data': self.knn_graph_data
        }, path)
        print(f"Codebook salvato in: {path}")
    
    def load_codebook(self, path: str):
        """
        Carica il codebook da disco.
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        self.codebook = nn.Parameter(checkpoint['codebook'])
        self.codebook_size = checkpoint['codebook_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.commitment_cost = checkpoint['commitment_cost']
        self.geodesic_distances = checkpoint.get('geodesic_distances', None)
        self.knn_graph_data = checkpoint.get('knn_graph_data', None)
        
        print(f"Codebook caricato da: {path}")
        print(f"  • Dimensione: {self.codebook_size}")
        print(f"  • Embedding dim: {self.embedding_dim}")
    
    def to(self, device):
        """Sposta il modulo sul dispositivo specificato."""
        super().to(device)
        self.codebook = self.codebook.to(device)
        return self


def test_geodesic_vector_quantizer():
    """
    Funzione di test per il GeodesicVectorQuantizer.
    """
    print("Test del GeodesicVectorQuantizer...")
    
    # Crea un quantizer di test
    quantizer = GeodesicVectorQuantizer(codebook_size=128, embedding_dim=32)
    
    # Carica dal file di clustering migliore
    clustering_file = "output/clustering_knn_graphs/clustering_32_18_128.npz"
    
    if Path(clustering_file).exists():
        quantizer.load_from_clustering_results(clustering_file)
        
        # Calcola distanze geodesiche tra i codici
        quantizer.compute_geodesic_distances_between_codes(k_neighbors=10)
        
        # Test della quantizzazione
        batch_size = 100
        test_vectors = torch.randn(batch_size, quantizer.embedding_dim)
        
        print(f"\nTest quantizzazione su {batch_size} vettori...")
        
        # Quantizzazione euclidea
        quantized_euc, loss_euc, indices_euc = quantizer.quantize(test_vectors, use_geodesic=False)
        
        # Quantizzazione geodesica
        quantized_geo, loss_geo, indices_geo = quantizer.quantize(test_vectors, use_geodesic=True)
        
        print(f"Loss euclidea: {loss_euc.item():.4f}")
        print(f"Loss geodesica: {loss_geo.item():.4f}")
        
        # Analizza l'utilizzo del codebook
        usage_euc = quantizer.get_codebook_usage(indices_euc)
        usage_geo = quantizer.get_codebook_usage(indices_geo)
        
        print(f"\nUtilizzo codebook (Euclidea): {usage_euc['used_codes']}/{usage_euc['total_codes']} ({usage_euc['usage_rate']:.2%})")
        print(f"Utilizzo codebook (Geodesica): {usage_geo['used_codes']}/{usage_geo['total_codes']} ({usage_geo['usage_rate']:.2%})")
        
        # Differenze nell'assegnazione
        different_assignments = (indices_euc != indices_geo).sum().item()
        print(f"Assegnazioni diverse tra euclidea e geodesica: {different_assignments}/{batch_size} ({different_assignments/batch_size:.1%})")
        
        # Salva il codebook
        quantizer.save_codebook("output/geodesic_codebook.pth")
        
        # Visualizza il codebook
        quantizer.visualize_codebook("output/codebook_visualization.png")
        
        print("\nTest completato con successo!")
        
    else:
        print(f"File di clustering non trovato: {clustering_file}")
        print("Esegui prima il training del clustering geodesico!")


if __name__ == "__main__":
    test_geodesic_vector_quantizer()
