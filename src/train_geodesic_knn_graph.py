import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import os
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

from models.vae import VAE
from models.knn_graph import KNNGraph
from models.kmeans_geodesic import KMeansGeodesic
from utils.dataloader import get_mnist_single_batch
from utils.device import device

import warnings
warnings.filterwarnings('ignore')

# Parametri fissi per gli esperimenti
RANDOM_STATE = 42


def load_trained_vae(model_path: str, latent_dim: int) -> VAE:
    """
    Carica un modello VAE pre-addestrato.
    """
    print(f"Caricamento modello VAE da: {model_path}")
    
    # Inizializza il modello
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
    
    # Carica i pesi
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"Modello VAE caricato (dimensione latente: {latent_dim})")
    return model


def extract_latent_representations(model: VAE, data: torch.Tensor):
    """
    Estrae le rappresentazioni latenti dal VAE da un singolo batch.
    """
    print(f"Estrazione rappresentazioni latenti da {len(data)} campioni...")
    
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        # Estrai la rappresentazione latente (media dell'encoder)
        mu, logvar = model.encode(data.view(-1, 784))
        latent_vectors = mu.cpu().numpy()
    
    print(f"Estratte rappresentazioni latenti di dimensione {latent_vectors.shape[1]}")
    return latent_vectors


def evaluate_clustering(true_labels, predicted_labels):
    """
    Valuta la qualità del clustering con diverse metriche.
    """
    metrics = {
        'ari': adjusted_rand_score(true_labels, predicted_labels),
        'nmi': normalized_mutual_info_score(true_labels, predicted_labels),
        'n_clusters': len(np.unique(predicted_labels))
    }
    return metrics


def save_visualization(
    latent_vectors,
    true_labels,
    geodesic_labels_dict, 
    output_path: str,
    config: dict,
    cluster_centers_dict: dict
):
    """
    Salva le visualizzazioni dei risultati del clustering per diversi n_clusters.
    Usa t-SNE 2D, colora i punti secondo le etichette vere MNIST, evidenzia i confini dei cluster geodesici.
    """
    # t-SNE per la visualizzazione 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Crea una figura con subplot multipli
    rows = math.ceil(len(geodesic_labels_dict) / 2)
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(
        f'Clustering Results: LD={config["latent_dim"]}, k={config["k_neighbors"]}',
        fontsize=16
    )
    
    # Assicurati che axes sia sempre un array 2D
    if rows == 1:
        axes = axes.reshape(1, -1)

    # Colori per le etichette MNIST (0-9)
    mnist_colors = plt.cm.tab10(np.arange(10))
    
    # Subplot per diversi n_clusters
    n_clusters_list = sorted(geodesic_labels_dict.keys())

    for i, n_clusters in enumerate(n_clusters_list):
        row, col = divmod(i, cols)
        cluster_centers = cluster_centers_dict[n_clusters]
        
        # Scatter plot principale colorato secondo le etichette vere MNIST
        for mnist_label in range(10):
            mask = true_labels == mnist_label
            if np.any(mask):
                axes[row, col].scatter(
                    latent_2d[mask, 0],
                    latent_2d[mask, 1],
                    c=[mnist_colors[mnist_label]],
                    label=f'MNIST {mnist_label}',
                    alpha=0.6,
                    s=15,
                    edgecolors='white',
                    linewidth=0.2
                )
        
        # Evidenzia i centroidi dei cluster geodesici
        for j, center_idx in enumerate(cluster_centers):
            # Il centroide è rappresentato dall'indice del punto più vicino al centro
            centroid_pos = latent_2d[center_idx]
            centroid_true_label = true_labels[center_idx]
            
            # Stella grande per il centroide
            axes[row, col].scatter(
                centroid_pos[0], centroid_pos[1],
                s=300,
                c=mnist_colors[centroid_true_label],
                marker='*',
                edgecolors='black',
                linewidth=2,
                alpha=0.9,
                zorder=5,
                label=f'Centroid' if j == 0 else None  # Label solo per il primo
            )
            
            # Cerchio attorno al centroide per evidenziarlo meglio
            circle = plt.Circle(centroid_pos, 0.5, fill=False, 
                              color=mnist_colors[centroid_true_label], alpha=0.7, linewidth=2, zorder=4)
            axes[row, col].add_patch(circle)
            
            # Annotazione con l'ID del cluster
            axes[row, col].annotate(
                f'C{j}', 
                centroid_pos, 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                color='red',
                zorder=6
            )
        
        axes[row, col].set_title(f"Geodesic K-Means (clusters={n_clusters})")
        axes[row, col].set_xlabel("t-SNE Dimension 1")
        axes[row, col].set_ylabel("t-SNE Dimension 2")
        axes[row, col].grid(True, alpha=0.3)
        
        # Legenda per le etichette MNIST (solo per il primo subplot)
        if i == 0:
            # Legenda MNIST in due colonne per compattezza
            mnist_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=mnist_colors[j], markersize=8, 
                                      label=f'MNIST {j}') for j in range(10)]
            centroid_handle = [plt.Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor='red', markersize=12, 
                                        markeredgecolor='black', label='Centroids')]
            
            all_handles = mnist_handles + centroid_handle
            axes[row, col].legend(handles=all_handles, loc='upper right', 
                                fontsize=8, ncol=2, framealpha=0.9)

    # Nascondi subplot vuoti se necessario
    for i in range(len(n_clusters_list), rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Salva l'immagine con nome specifico
    img_path = output_path.replace('.npz', '.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizzazione salvata: {img_path}")


def run_clustering_experiments(
    latent_dim: int,
    k_neighbors: int,
    n_clusters_options: list, 
    data: torch.Tensor,
    labels: torch.Tensor
):
    """
    Esegue esperimenti di clustering geodesico per diversi valori di n_clusters con grafo condiviso.
    """
    print(f"\n{'='*80}")
    print(f"ESPERIMENTI: LD={latent_dim}, k={k_neighbors}")
    print(f"{'='*80}")

    start_time = time.time()

    # Configura il modello VAE
    model_path = f'output/vae/vae_model_{latent_dim}d.pth'

    # Verifica che il modello esista
    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}")
        return None

    # Carica il modello VAE
    vae_model = load_trained_vae(model_path, latent_dim)

    # Estrai rappresentazioni latenti
    latent_vectors = extract_latent_representations(vae_model, data)
    true_labels = labels.numpy()

    # Crea directory output per questo k_neighbors
    output_dir = Path(f'output/clustering_knn_graphs/k_{k_neighbors}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva il grafo e le distanze geodesiche
    graph_data_path = output_dir / f'knn_graph_k_{k_neighbors}.npz'
    
    # Inizializza il grafo k-NN geodesico (condiviso per tutti i clustering)
    knn_graph = KNNGraph(k=k_neighbors)
    
    # Prova a caricare il grafo esistente
    if knn_graph.load_graph(str(graph_data_path)):
        print(f"Grafo esistente caricato da: {graph_data_path}")
        # Verifica che i dati siano compatibili
        if not np.array_equal(knn_graph.data_points, latent_vectors):
            print("Dati non compatibili, ricostruendo il grafo...")
            knn_graph.fit(latent_vectors, metric='euclidean')
            knn_graph.save_graph(str(graph_data_path))
    else:
        print(f"Costruzione grafo k-NN con k={k_neighbors} e calcolo delle distanze geodesiche...")
        knn_graph.fit(latent_vectors, metric='euclidean')
        knn_graph.save_graph(str(graph_data_path))

    # Dizionari per raccogliere risultati
    all_geodesic_labels = {}
    all_geodesic_metrics = {}
    all_cluster_centers = {}

    # Esegui clustering per ogni n_clusters
    for n_clusters in n_clusters_options:
        print(f"\nEsecuzione clustering K-Means Geodesico con {n_clusters} clusters...")
        
        kmeans_geodesic = KMeansGeodesic(
            knn_graph=knn_graph.graph, 
            distance_matrix=knn_graph.geodesic_distances, 
            n_clusters=n_clusters, 
            random_state=RANDOM_STATE
        )
        
        # Conversione dei samples in indici
        sample_indices = np.arange(latent_vectors.shape[0])
        kmeans_geodesic.fit(sample_indices)
        
        # Salva risultati
        all_geodesic_labels[n_clusters] = kmeans_geodesic.labels_
        all_cluster_centers[n_clusters] = kmeans_geodesic.cluster_centers_
        
        # Valuta i risultati
        geodesic_metrics = evaluate_clustering(true_labels, kmeans_geodesic.labels_)
        all_geodesic_metrics[n_clusters] = geodesic_metrics
        
        print(f'Clustering K-Means Geodesico ({n_clusters} clusters) completato.')
        print(f"  ◦ ARI: {geodesic_metrics['ari']:.3f}")
        print(f"  ◦ NMI: {geodesic_metrics['nmi']:.3f}")

        # Salva risultati individuali del clustering (sovrascrive se esiste)
        cluster_filename = f'clustering_k_{k_neighbors}_clusters_{n_clusters}.npz'
        cluster_path = output_dir / cluster_filename
        
        config = {
            'latent_dim': latent_dim,
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'max_samples': len(data),
            'random_state': RANDOM_STATE
        }

        # Usa numpy per sovrascrivere il file se esiste
        if cluster_path.exists():
            print(f"Sovrascrittura file esistente: {cluster_path}")
        
        np.savez_compressed(
            str(cluster_path),
            latent_vectors=latent_vectors,
            true_labels=true_labels,
            geodesic_labels=kmeans_geodesic.labels_,
            geodesic_centers=kmeans_geodesic.cluster_centers_,
            geodesic_metrics=geodesic_metrics,
            config=config
        )

    # Salva visualizzazione con tutti i n_clusters
    print("Generazione visualizzazioni...")
    plot_filename = f'clustering_k_{k_neighbors}_comparison.png'
    plot_path = output_dir / plot_filename
    
    config = {
        'latent_dim': latent_dim,
        'k_neighbors': k_neighbors,
        'n_clusters_options': n_clusters_options
    }
    
    save_visualization(
        latent_vectors, true_labels, 
        all_geodesic_labels,
        str(plot_path), config,
        all_cluster_centers
    )

    elapsed_time = time.time() - start_time

    # Mostra analisi risultati
    print(f"\nANALISI RISULTATI:")
    print(f"  Tempo impiegato: {elapsed_time:.1f}s")
    print(f"  Campioni processati: {len(latent_vectors)}")
    print(f"  Dimensione spazio latente: {latent_vectors.shape[1]}")
    print(f"  k-neighbors: {k_neighbors}")

    print(f"\nMETRICHE DI CLUSTERING:")
    for n_clusters in n_clusters_options:
        metrics = all_geodesic_metrics[n_clusters]
        print(f"  {n_clusters} clusters:")
        print(f"    ◦ ARI: {metrics['ari']:.3f}")
        print(f"    ◦ NMI: {metrics['nmi']:.3f}")

    print(f"\nRisultati salvati in: {output_dir}")

    # Ritorna i risultati per il riepilogo finale
    return {
        'k_neighbors': k_neighbors,
        'latent_dim': latent_dim,
        'all_geodesic_metrics': all_geodesic_metrics,
        'n_clusters_options': n_clusters_options,
        'elapsed_time': elapsed_time,
        'output_dir': str(output_dir)
    }


def show_final_summary(all_results):
    """
    Mostra un riepilogo finale di tutti i risultati.
    """
    print(f"\n{'='*100}")
    print(f"RIEPILOGO FINALE - {len(all_results)} ESPERIMENTI COMPLETATI")
    print(f"{'='*100}")
    
    # Tabella riassuntiva
    print(f"\nTABELLA RIASSUNTIVA:")
    print(f"{'k-neighbors':<12} {'n_clusters':<12} {'ARI':<8} {'NMI':<8} {'Tempo':<8} {'Output Dir':<30}")
    print("-" * 100)
    
    for result in all_results:
        k_neighbors = result['k_neighbors']
        for n_clusters in result['n_clusters_options']:
            metrics = result['all_geodesic_metrics'][n_clusters]
            print(f"{k_neighbors:<12} "
                  f"{n_clusters:<12} "
                  f"{metrics['ari']:<8.3f} "
                  f"{metrics['nmi']:<8.3f} "
                  f"{result['elapsed_time']:<8.1f} "
                  f"{result['output_dir']:<30}")
    
    # Statistiche aggregate per k-neighbors
    print(f"\nSTATISTICHE PER k-NEIGHBORS:")
    for result in all_results:
        k_neighbors = result['k_neighbors']
        ari_values = [result['all_geodesic_metrics'][n]['ari'] for n in result['n_clusters_options']]
        nmi_values = [result['all_geodesic_metrics'][n]['nmi'] for n in result['n_clusters_options']]
        
        print(f"  k={k_neighbors}:")
        print(f"    ARI medio: {np.mean(ari_values):.3f} (std: {np.std(ari_values):.3f})")
        print(f"    NMI medio: {np.mean(nmi_values):.3f} (std: {np.std(nmi_values):.3f})")
    
    total_time = sum(r['elapsed_time'] for r in all_results)
    print(f"\nTempo totale: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    print(f"\nFile salvati per k-neighbors:")
    for result in all_results:
        print(f"  k={result['k_neighbors']}: {result['output_dir']}")


def main():
    """
    Funzione principale che esegue tutti i loop di esperimenti.
    """
    print("AVVIO ESPERIMENTI CLUSTERING GEODESICO K-MEANS")
    print("="*80)

    # Parametri degli esperimenti
    latent_dim = 32
    k_neighbors_range = [8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 25]
    n_clusters_options = [64, 96, 128, 256, 512]

    # Setup device
    print(f"Device: {device}")

    # Carica i dati una sola volta
    print(f"\nCaricamento dataset MNIST...")
    data, labels = get_mnist_single_batch(max_samples=10000, split="test")

    # Log dei parametri degli esperimenti
    print(f"CONFIGURAZIONE ESPERIMENTI:")
    print(f"  • Dimensione latente fissa: {latent_dim}")
    print(f"  • k-neighbors: {k_neighbors_range}")
    print(f"  • n_clusters: {n_clusters_options}")
    print(f"  • Campioni fissi: {len(data)}")
    print(f"  • Totale esperimenti k-neighbors: {len(k_neighbors_range)}")
    print(f"  • Clustering per k-neighbors: {len(n_clusters_options)}")

    # Esegui tutti gli esperimenti
    all_results = []
    current_experiment = 0

    for k_neighbors in k_neighbors_range:
        current_experiment += 1
        print(f"\nProgresso: {current_experiment}/{len(k_neighbors_range)}")

        try:
            result = run_clustering_experiments(latent_dim, k_neighbors, n_clusters_options, data, labels)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"Errore nell'esperimento k={k_neighbors}: {e}")
            continue

    # Mostra riepilogo finale
    if all_results:
        show_final_summary(all_results)
    else:
        print("Nessun esperimento completato con successo!")


if __name__ == "__main__":
    main()
