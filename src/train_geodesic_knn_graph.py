import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
from itertools import product

from models.vae import VAE
from models.knn_graph import KNNGraph
from models.kmeans_euclidean import KMeansEuclidean
from models.kmeans_geodesic import KMeansGeodesic
from utils.dataloader import get_mnist_single_batch
from utils.device import device
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

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


def save_visualization(latent_vectors, true_labels, euclidean_labels, geodesic_labels, 
                      output_path: str, config: dict):
    """
    Salva le visualizzazioni dei risultati del clustering.
    """
    # t-SNE per la visualizzazione 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Crea una figura con subplot multipli
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Clustering Results: LD={config["latent_dim"]}, k={config["k_neighbors"]}, clusters={config["n_clusters"]}', 
                 fontsize=16)
    
    # 1. Etichette vere
    scatter = axes[0, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=true_labels, 
                                cmap='tab10', alpha=0.7, s=2)
    axes[0, 0].set_title('Etichette Vere (MNIST)')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. K-Means Euclideo
    scatter = axes[0, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=euclidean_labels, 
                                cmap='tab10', alpha=0.7, s=2)
    axes[0, 1].set_title('K-Means Euclideo')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. K-Means Geodesico
    scatter = axes[1, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=geodesic_labels, 
                                cmap='tab10', alpha=0.7, s=2)
    axes[1, 0].set_title('K-Means Geodesico (k-NN Graph)')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Differenze tra i due metodi
    differences = (euclidean_labels != geodesic_labels).astype(int)
    scatter = axes[1, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=differences, 
                                cmap='RdYlBu', alpha=0.7, s=2)
    axes[1, 1].set_title('Differenze tra Euclideo e Geodesico')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Salva l'immagine con nome specifico
    img_path = output_path.replace('.npz', '.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()  # Chiudi la figura per risparmiare memoria
    
    print(f"Visualizzazione salvata: {img_path}")


def run_single_experiment(latent_dim: int, k_neighbors: int, n_clusters: int, 
                         data: torch.Tensor, labels: torch.Tensor):
    """
    Esegue un singolo esperimento di clustering geodesico.
    """
    print(f"\n{'='*80}")
    print(f"ESPERIMENTO: LD={latent_dim}, k={k_neighbors}, clusters={n_clusters}")
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

    # Inizializza il grafo k-NN geodesico
    print(f"Costruzione grafo k-NN con k={k_neighbors} e calcolo delle distanze geodesiche...")
    knn_graph = KNNGraph(k=k_neighbors)

    # Costruisci il grafo k-NN e calcola le distanze
    knn_graph.fit(latent_vectors, metric='euclidean')

    # Esecuzione clustering K-Means con metrica euclidea
    print("Esecuzione clustering K-Means Euclideo...")
    kmeans_euclidean = KMeansEuclidean(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans_euclidean.fit(latent_vectors)
    print('Clustering K-Means Euclideo completato.')

    # Esecuzione clustering K-Means con metrica geodesica
    print("Esecuzione clustering K-Means Geodesico...")
    kmeans_geodesic = KMeansGeodesic(
        knn_graph=knn_graph.graph, 
        distance_matrix=knn_graph.geodesic_distances, 
        n_clusters=n_clusters, 
        random_state=RANDOM_STATE
    )
    # Conversione dei samples in indici
    sample_indices = np.arange(latent_vectors.shape[0])
    kmeans_geodesic.fit(sample_indices)
    print('Clustering K-Means Geodesico completato.')

    # Calcola i risultati di confronto
    comparison_results = {
        "euclidean_labels": kmeans_euclidean.euclidean_labels,
        "geodesic_labels": kmeans_geodesic.labels_,
        "geodesic_centers": kmeans_geodesic.cluster_centers_,
        "euclidean_centers": kmeans_euclidean.kmeans_alg.cluster_centers_,
        "euclidean_inertia": kmeans_euclidean.kmeans_alg.inertia_,
        "n_different_assignments": np.sum(
            kmeans_euclidean.euclidean_labels != kmeans_geodesic.labels_
        ),
        "agreement_percentage": np.mean(
            kmeans_euclidean.euclidean_labels == kmeans_geodesic.labels_
        )
        * 100,
    }

    print(f"Accordo tra clustering euclideo e geodesico: {comparison_results['agreement_percentage']:.1f}%")
    print(f"Punti assegnati diversamente: {comparison_results['n_different_assignments']}")

    # Valuta i risultati
    euclidean_metrics = evaluate_clustering(true_labels, comparison_results['euclidean_labels'])
    geodesic_metrics = evaluate_clustering(true_labels, comparison_results['geodesic_labels'])

    # Crea directory output se non esiste
    output_dir = Path('output/clustering_knn_graphs')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva i risultati
    output_filename = f'clustering_{latent_dim}_{k_neighbors}_{n_clusters}.npz'
    output_path = output_dir / output_filename

    config = {
        'latent_dim': latent_dim,
        'k_neighbors': k_neighbors,
        'n_clusters': n_clusters,
        'max_samples': len(data),
        'random_state': RANDOM_STATE
    }

    np.savez(
        str(output_path),
        latent_vectors=latent_vectors,
        true_labels=true_labels,
        euclidean_labels=comparison_results['euclidean_labels'],
        geodesic_labels=comparison_results['geodesic_labels'],
        geodesic_centers=comparison_results['geodesic_centers'],
        euclidean_metrics=euclidean_metrics,
        geodesic_metrics=geodesic_metrics,
        comparison_results=comparison_results,
        config=config
    )

    # Salva visualizzazione
    print("Generazione visualizzazioni...")
    save_visualization(
        latent_vectors, true_labels, 
        comparison_results['euclidean_labels'], 
        comparison_results['geodesic_labels'],
        str(output_path), config
    )

    elapsed_time = time.time() - start_time

    # Mostra analisi risultati
    print(f"\nANALISI RISULTATI:")
    print(f"  Tempo impiegato: {elapsed_time:.1f}s")
    print(f"  Campioni processati: {len(latent_vectors)}")
    print(f"  Dimensione spazio latente: {latent_vectors.shape[1]}")

    print(f"\nMETRICHE DI CLUSTERING:")
    print(f"  K-Means Euclideo:")
    print(f"    ◦ ARI: {euclidean_metrics['ari']:.3f}")
    print(f"    ◦ NMI: {euclidean_metrics['nmi']:.3f}")

    print(f"  K-Means Geodesico:")
    print(f"    ◦ ARI: {geodesic_metrics['ari']:.3f}")
    print(f"    ◦ NMI: {geodesic_metrics['nmi']:.3f}")

    print(f"\nCONFRONTO TRA METODI:")
    print(f"  • Accordo: {comparison_results['agreement_percentage']:.1f}%")
    print(f"  • Punti assegnati diversamente: {comparison_results['n_different_assignments']}")

    ari_improvement = geodesic_metrics['ari'] - euclidean_metrics['ari']
    nmi_improvement = geodesic_metrics['nmi'] - euclidean_metrics['nmi']
    print(f"  • Miglioramento ARI: {ari_improvement:+.3f}")
    print(f"  • Miglioramento NMI: {nmi_improvement:+.3f}")

    # Interpretazione
    if ari_improvement > 0.05:
        interpretation = "Geodesico significativamente migliore"
    elif ari_improvement > 0:
        interpretation = "Geodesico leggermente migliore"
    elif ari_improvement > -0.05:
        interpretation = "≈ Metodi comparabili"
    else:
        interpretation = "Euclideo migliore"

    print(f"  Interpretazione: {interpretation}")

    print(f"\nRisultati salvati in: {output_path}")

    # Ritorna i risultati per il riepilogo finale
    return {
        'config': config,
        'euclidean_metrics': euclidean_metrics,
        'geodesic_metrics': geodesic_metrics,
        'comparison_results': comparison_results,
        'ari_improvement': ari_improvement,
        'nmi_improvement': nmi_improvement,
        'elapsed_time': elapsed_time,
        'interpretation': interpretation
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
    print(f"{'Config':<20} {'ARI_Euc':<8} {'ARI_Geo':<8} {'Δ_ARI':<8} {'NMI_Euc':<8} {'NMI_Geo':<8} {'Δ_NMI':<8} {'Accordo%':<10} {'Tempo':<8}")
    print("-" * 100)
    
    for result in all_results:
        config = result['config']
        config_str = f"{config['latent_dim']},{config['k_neighbors']},{config['n_clusters']}"
        
        print(f"{config_str:<20} "
              f"{result['euclidean_metrics']['ari']:<8.3f} "
              f"{result['geodesic_metrics']['ari']:<8.3f} "
              f"{result['ari_improvement']:<8.3f} "
              f"{result['euclidean_metrics']['nmi']:<8.3f} "
              f"{result['geodesic_metrics']['nmi']:<8.3f} "
              f"{result['nmi_improvement']:<8.3f} "
              f"{result['comparison_results']['agreement_percentage']:<10.1f} "
              f"{result['elapsed_time']:<8.1f}")
    
    # Statistiche aggregate
    ari_improvements = [r['ari_improvement'] for r in all_results]
    nmi_improvements = [r['nmi_improvement'] for r in all_results]
    
    print(f"\nSTATISTICHE AGGREGATE:")
    print(f"  Miglioramento ARI medio: {np.mean(ari_improvements):+.3f} (std: {np.std(ari_improvements):.3f})")
    print(f"  Miglioramento NMI medio: {np.mean(nmi_improvements):+.3f} (std: {np.std(nmi_improvements):.3f})")
    
    # Migliori configurazioni
    best_ari_idx = np.argmax(ari_improvements)
    best_nmi_idx = np.argmax(nmi_improvements)
    
    print(f"\nMIGLIORI CONFIGURAZIONI:")
    
    best_ari_config = all_results[best_ari_idx]['config']
    print(f"  Miglior ARI: LD={best_ari_config['latent_dim']}, k={best_ari_config['k_neighbors']}, "
          f"clusters={best_ari_config['n_clusters']} (Δ={ari_improvements[best_ari_idx]:+.3f})")
    
    best_nmi_config = all_results[best_nmi_idx]['config']
    print(f"  Miglior NMI: LD={best_nmi_config['latent_dim']}, k={best_nmi_config['k_neighbors']}, "
          f"clusters={best_nmi_config['n_clusters']} (Δ={nmi_improvements[best_nmi_idx]:+.3f})")
    
    # Conteggio interpretazioni
    interpretations = [r['interpretation'] for r in all_results]
    print(f"\nDISTRIBUZIONE RISULTATI:")
    for interp in set(interpretations):
        count = interpretations.count(interp)
        print(f"  {interp}: {count}/{len(all_results)} esperimenti")
    
    total_time = sum(r['elapsed_time'] for r in all_results)
    print(f"\nTempo totale: {total_time:.1f}s ({total_time/60:.1f}m)")


def main():
    """
    Funzione principale che esegue tutti i loop di esperimenti.
    """
    print("AVVIO ESPERIMENTI CLUSTERING GEODESICO K-MEANS")
    print("="*80)

    # Parametri degli esperimenti
    latent_dims = [32]
    k_neighbors_range = [8, 16, 24, 32]
    n_clusters_options = [10]

    # Setup device
    print(f"Device: {device}")

    # Carica i dati una sola volta
    print(f"\nCaricamento dataset MNIST...")
    data, labels = get_mnist_single_batch(max_samples=10000, split="test")

    # Log dei parametri degli esperimenti
    print(f"CONFIGURAZIONE ESPERIMENTI:")
    print(f"  • Dimensioni latenti: {latent_dims}")
    print(f"  • k-neighbors: {k_neighbors_range}")
    print(f"  • n_clusters: {n_clusters_options}")
    print(f"  • Campioni fissi: {len(data)}")
    print(
        f"  • Totale esperimenti: {len(latent_dims) * len(k_neighbors_range) * len(n_clusters_options)}"
    )

    # Esegui tutti gli esperimenti
    all_results = []
    total_experiments = len(latent_dims) * len(k_neighbors_range) * len(n_clusters_options)
    current_experiment = 0

    for latent_dim, k_neighbors, n_clusters in product(latent_dims, k_neighbors_range, n_clusters_options):
        current_experiment += 1
        print(f"\nProgresso: {current_experiment}/{total_experiments}")

        try:
            result = run_single_experiment(latent_dim, k_neighbors, n_clusters, data, labels)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"Errore nell'esperimento LD={latent_dim}, k={k_neighbors}, clusters={n_clusters}: {e}")
            continue

    # Mostra riepilogo finale
    if all_results:
        show_final_summary(all_results)
    else:
        print("Nessun esperimento completato con successo!")


if __name__ == "__main__":
    main()
