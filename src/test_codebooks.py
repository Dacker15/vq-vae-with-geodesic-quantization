import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import time
import argparse

from models.vae import VAE
from models.knn_graph import KNNGraph
from utils.dataloader import get_mnist_single_batch
from utils.device import device

import warnings
warnings.filterwarnings('ignore')


class GeodeticQuantizer:
    """
    Classe per effettuare quantizzazione geodesica usando i codebook creati.
    """
    
    def __init__(self, codebooks_base_dir: str = 'output/codebooks', 
                 clustering_base_dir: str = 'output/clustering_knn_graphs'):
        self.codebooks_base_dir = Path(codebooks_base_dir)
        self.clustering_base_dir = Path(clustering_base_dir)
        self.loaded_codebooks = {}
        self.loaded_graphs = {}
        self.loaded_vaes = {}
        
    def load_vae_model(self, latent_dim: int) -> VAE:
        """
        Carica un modello VAE pre-addestrato.
        
        Args:
            latent_dim: Dimensione dello spazio latente
            
        Returns:
            Modello VAE caricato
        """
        if latent_dim in self.loaded_vaes:
            return self.loaded_vaes[latent_dim]
        
        model_path = f'output/vae/vae_model_{latent_dim}d.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello VAE non trovato: {model_path}")
        
        print(f"Caricamento modello VAE da: {model_path}")
        
        # Inizializza il modello
        model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
        
        # Carica i pesi
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        self.loaded_vaes[latent_dim] = model
        print(f"Modello VAE caricato (dimensione latente: {latent_dim})")
        
        return model
    
    def load_codebook(self, k_neighbors: int, n_clusters: int) -> Dict:
        """
        Carica un codebook specifico.
        
        Args:
            k_neighbors: Valore k del grafo k-NN
            n_clusters: Numero di cluster
            
        Returns:
            Dizionario con i dati del codebook
        """
        key = f"k_{k_neighbors}_clusters_{n_clusters}"
        
        if key in self.loaded_codebooks:
            return self.loaded_codebooks[key]
        
        codebook_path = self.codebooks_base_dir / f'k_{k_neighbors}' / f'codebook_k_{k_neighbors}_clusters_{n_clusters}.npz'
        
        if not codebook_path.exists():
            raise FileNotFoundError(f"Codebook non trovato: {codebook_path}")
        
        try:
            data = np.load(codebook_path, allow_pickle=True)
            
            codebook_data = {
                'codebook': data['codebook'],
                'codebook_info': data['codebook_info'].item(),
                'original_config': data['original_config'].item(),
                'geodesic_centers_indices': data['geodesic_centers_indices'],
                'geodesic_labels': data['geodesic_labels'],
                'file_path': str(codebook_path)
            }
            
            self.loaded_codebooks[key] = codebook_data
            print(f"Codebook caricato: k={k_neighbors}, clusters={n_clusters}, shape={codebook_data['codebook'].shape}")
            
            return codebook_data
            
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento del codebook {codebook_path}: {e}")
    
    def load_knn_graph(self, k_neighbors: int) -> KNNGraph:
        """
        Carica il grafo k-NN pre-calcolato.
        
        Args:
            k_neighbors: Valore k del grafo k-NN
            
        Returns:
            Oggetto KNNGraph caricato
        """
        if k_neighbors in self.loaded_graphs:
            return self.loaded_graphs[k_neighbors]
        
        graph_path = self.clustering_base_dir / f'k_{k_neighbors}' / f'knn_graph_k_{k_neighbors}.npz'
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Grafo k-NN non trovato: {graph_path}")
        
        knn_graph = KNNGraph(k=k_neighbors)
        
        if not knn_graph.load_graph(str(graph_path)):
            raise RuntimeError(f"Impossibile caricare il grafo da {graph_path}")
        
        self.loaded_graphs[k_neighbors] = knn_graph
        print(f"Grafo k-NN caricato: k={k_neighbors}")
        
        return knn_graph
    
    def encode_sample(self, vae_model: VAE, sample: torch.Tensor) -> np.ndarray:
        """
        Codifica un campione usando il VAE.
        
        Args:
            vae_model: Modello VAE
            sample: Campione da codificare [1, 28, 28] o [784]
            
        Returns:
            Vettore latente [D,]
        """
        vae_model.eval()
        with torch.no_grad():
            sample = sample.to(device)
            if sample.dim() == 3:  # [1, 28, 28]
                sample = sample.view(1, -1)  # [1, 784]
            elif sample.dim() == 2:  # [28, 28]
                sample = sample.view(1, -1)  # [1, 784]
            elif sample.dim() == 1:  # [784]
                sample = sample.view(1, -1)  # [1, 784]
            
            mu, logvar = vae_model.encode(sample)
            latent_vector = mu.cpu().numpy().flatten()  # [D,]
        
        return latent_vector
    
    def quantize_with_geodesic_distance(self, latent_vector: np.ndarray, 
                                      knn_graph: KNNGraph, 
                                      cluster_centers_indices: np.ndarray) -> Tuple[int, int]:
        """
        Quantizza un vettore latente usando le distanze geodesiche.
        
        Args:
            latent_vector: Vettore latente da quantizzare [D,]
            knn_graph: Grafo k-NN pre-calcolato
            cluster_centers_indices: Indici dei centroidi dei cluster
            
        Returns:
            Tuple (cluster_id, centroid_index) dove:
            - cluster_id: ID del cluster più vicino
            - centroid_index: Indice del centroide nel grafo originale
        """
        # Aggiungi il nuovo punto al grafo
        new_node_idx = knn_graph.add_node_to_graph(latent_vector)
        
        # Calcola le distanze geodesiche dal nuovo punto ai centroidi
        geodesic_distances = knn_graph.get_geodesic_distance_to_points(
            new_node_idx, cluster_centers_indices
        )
        
        # Trova il centroide più vicino
        closest_cluster_id = np.argmin(geodesic_distances)
        closest_centroid_index = cluster_centers_indices[closest_cluster_id]
        
        return closest_cluster_id, closest_centroid_index
    
    def decode_centroid(self, vae_model: VAE, centroid_vector: np.ndarray) -> np.ndarray:
        """
        Decodifica un vettore centroide usando il VAE.
        
        Args:
            vae_model: Modello VAE
            centroid_vector: Vettore centroide [D,]
            
        Returns:
            Immagine ricostruita [28, 28]
        """
        vae_model.eval()
        with torch.no_grad():
            centroid_tensor = torch.tensor(centroid_vector, dtype=torch.float32).unsqueeze(0).to(device)  # [1, D]
            reconstructed = vae_model.decode(centroid_tensor)
            reconstructed_image = reconstructed.cpu().numpy().reshape(28, 28)
        
        return reconstructed_image
    
    def full_quantization_pipeline(self, sample: torch.Tensor, k_neighbors: int, 
                                 n_clusters: int) -> Dict:
        """
        Esegue il pipeline completo di quantizzazione geodesica.
        
        Args:
            sample: Campione MNIST [1, 28, 28] o [28, 28]
            k_neighbors: Valore k del grafo k-NN
            n_clusters: Numero di cluster
            
        Returns:
            Dizionario con tutti i risultati del pipeline
        """
        # Carica i componenti necessari
        codebook_data = self.load_codebook(k_neighbors, n_clusters)
        knn_graph = self.load_knn_graph(k_neighbors)
        latent_dim = codebook_data['original_config']['latent_dim']
        vae_model = self.load_vae_model(latent_dim)
        
        # 1. Encoding
        latent_vector = self.encode_sample(vae_model, sample)
        
        # 2. Quantizzazione geodesica
        cluster_id, centroid_index = self.quantize_with_geodesic_distance(
            latent_vector, knn_graph, codebook_data['geodesic_centers_indices']
        )
        
        # 3. Ottieni il vettore centroide dal codebook
        centroid_vector = codebook_data['codebook'][cluster_id]
        
        # 4. Decodifica
        reconstructed_image = self.decode_centroid(vae_model, centroid_vector)
        
        # 5. Decodifica anche del vettore originale per confronto
        original_reconstructed = self.decode_centroid(vae_model, latent_vector)
        
        return {
            'original_sample': sample.cpu().numpy() if isinstance(sample, torch.Tensor) else sample,
            'latent_vector': latent_vector,
            'cluster_id': cluster_id,
            'centroid_index': centroid_index,
            'centroid_vector': centroid_vector,
            'reconstructed_image': reconstructed_image,
            'original_reconstructed': original_reconstructed,
            'config': {
                'k_neighbors': k_neighbors,
                'n_clusters': n_clusters,
                'latent_dim': latent_dim
            }
        }


def visualize_quantization_results(results: List[Dict], save_path: Optional[str] = None):
    """
    Visualizza i risultati della quantizzazione con un campione per ogni cifra (0-9).
    
    Args:
        results: Lista dei risultati della quantizzazione (uno per cifra)
        save_path: Percorso per salvare la visualizzazione (opzionale)
    """
    # Organizza i risultati per cifra
    results_by_digit = {}
    for result in results:
        digit = result['true_label']
        results_by_digit[digit] = result
    
    # Assicurati di avere risultati per tutte le cifre 0-9
    n_digits = len(results_by_digit)
    if n_digits == 0:
        print("Nessun risultato da visualizzare")
        return
    
    fig, axes = plt.subplots(3, n_digits, figsize=(2 * n_digits, 8))
    
    if n_digits == 1:
        axes = axes.reshape(-1, 1)
    
    # Ordina per cifra
    sorted_digits = sorted(results_by_digit.keys())
    
    for i, digit in enumerate(sorted_digits):
        result = results_by_digit[digit]
        
        # Riga 1: Immagine originale
        axes[0, i].imshow(result['original_sample'].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Originale\nCifra: {digit}')
        axes[0, i].axis('off')
        
        # Riga 2: Ricostruzione con vettore originale
        axes[1, i].imshow(result['original_reconstructed'], cmap='gray')
        axes[1, i].set_title(f'Ricostr. Diretta\n(VAE)')
        axes[1, i].axis('off')
        
        # Riga 3: Ricostruzione con quantizzazione geodesica
        axes[2, i].imshow(result['reconstructed_image'], cmap='gray')
        config = result['config']
        axes[2, i].set_title(f'Ricostr. Quantizzata\nCluster: {result["cluster_id"]}')
        axes[2, i].axis('off')
    
    # Titolo generale
    if results:
        config = results[0]['config']
        fig.suptitle(f'Quantizzazione Geodesica: k={config["k_neighbors"]}, clusters={config["n_clusters"]}, '
                    f'latent_dim={config["latent_dim"]}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizzazione salvata: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_fidelity_metrics(quantizer: 'GeodeticQuantizer', k_neighbors: int, 
                             n_clusters: int, n_samples: int = 128) -> Dict:
    """
    Calcola metriche di fedeltà per un codebook su un campione di test.
    
    Args:
        quantizer: Oggetto GeodeticQuantizer
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        n_samples: Numero di campioni da testare
        
    Returns:
        Dizionario con metriche di fedeltà
    """
    print(f"Calcolo metriche di fedeltà con {n_samples} campioni...")
    
    # Carica campioni casuali dal test set
    data, labels = get_mnist_single_batch(max_samples=n_samples * 2, split="test")
    
    # Seleziona campioni casuali
    np.random.seed(42)  # Per riproducibilità
    random_indices = np.random.choice(len(data), n_samples, replace=False)
    test_samples = data[random_indices]
    test_labels = labels[random_indices]
    
    # Metriche di ricostruzione
    mse_values = []
    ssim_values = []
    cluster_assignments = []
    processing_times = []
    
    # Contatori per analisi di classe
    class_cluster_assignments = {i: [] for i in range(10)}
    
    for i, (sample, true_label) in enumerate(zip(test_samples, test_labels)):
        if i % 20 == 0:
            print(f"  Processando campione {i+1}/{n_samples}...")
        
        try:
            start_time = time.time()
            result = quantizer.full_quantization_pipeline(sample, k_neighbors, n_clusters)
            processing_time = time.time() - start_time
            
            # Calcola MSE tra originale e ricostruito
            original_flat = result['original_sample'].flatten()
            reconstructed_flat = result['reconstructed_image'].flatten()
            mse = np.mean((original_flat - reconstructed_flat) ** 2)
            
            # Calcola SSIM (struttura similarity index) - implementazione semplificata
            def simple_ssim(img1, img2):
                """Implementazione semplificata di SSIM."""
                # Normalizza le immagini
                img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
                img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
                
                # Calcola correlazione
                correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
                return max(0, correlation)  # Assicura che sia non negativo
            
            ssim_score = simple_ssim(result['original_sample'].reshape(28, 28), 
                                   result['reconstructed_image'])
            
            mse_values.append(mse)
            ssim_values.append(ssim_score)
            cluster_assignments.append(result['cluster_id'])
            processing_times.append(processing_time)
            
            # Analisi per classe
            true_label_int = true_label.item()
            class_cluster_assignments[true_label_int].append(result['cluster_id'])
            
        except Exception as e:
            print(f"  Errore nel campione {i}: {e}")
            continue
    
    # Calcola statistiche generali
    avg_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    avg_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    avg_time = np.mean(processing_times)
    
    # Analisi cluster utilizzati
    unique_clusters = len(set(cluster_assignments))
    cluster_usage = np.bincount(cluster_assignments, minlength=n_clusters)
    cluster_utilization = unique_clusters / n_clusters
    
    # Analisi consistenza per classe (stesso cluster per stessa cifra)
    class_consistency = {}
    for digit in range(10):
        assignments = class_cluster_assignments[digit]
        if len(assignments) > 0:
            # Calcola quanto spesso viene assegnato lo stesso cluster principale
            most_common_cluster = max(set(assignments), key=assignments.count)
            consistency = assignments.count(most_common_cluster) / len(assignments)
            class_consistency[digit] = {
                'consistency': consistency,
                'most_common_cluster': most_common_cluster,
                'total_samples': len(assignments),
                'unique_clusters': len(set(assignments))
            }
        else:
            class_consistency[digit] = {
                'consistency': 0.0,
                'most_common_cluster': -1,
                'total_samples': 0,
                'unique_clusters': 0
            }
    
    # Consistenza media
    valid_consistencies = [v['consistency'] for v in class_consistency.values() if v['total_samples'] > 0]
    avg_class_consistency = np.mean(valid_consistencies) if valid_consistencies else 0.0
    
    return {
        'reconstruction_quality': {
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'avg_ssim': avg_ssim,
            'std_ssim': std_ssim
        },
        'cluster_analysis': {
            'unique_clusters_used': unique_clusters,
            'total_clusters': n_clusters,
            'cluster_utilization': cluster_utilization,
            'cluster_usage_distribution': cluster_usage.tolist()
        },
        'class_consistency': class_consistency,
        'avg_class_consistency': avg_class_consistency,
        'performance': {
            'avg_processing_time': avg_time,
            'total_samples_processed': len(mse_values)
        },
        'config': {
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
    }


def test_single_codebook(quantizer: 'GeodeticQuantizer', k_neighbors: int, n_clusters: int, overwrite: bool = False) -> Dict:
    """
    Testa un singolo codebook con tutte le analisi richieste.
    
    Args:
        quantizer: Oggetto GeodeticQuantizer
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        overwrite: Se True, sovrascrive i test esistenti
        
    Returns:
        Dizionario con tutti i risultati del test
    """
    print(f"\n{'='*60}")
    print(f"TEST CODEBOOK: k={k_neighbors}, clusters={n_clusters}")
    print(f"{'='*60}")
    
    # Controlla se i file esistono già
    viz_path = quantizer.codebooks_base_dir / f'k_{k_neighbors}' / f'demo_quantization_k{k_neighbors}_c{n_clusters}.png'
    results_path = quantizer.codebooks_base_dir / f'k_{k_neighbors}' / f'test_results_k{k_neighbors}_c{n_clusters}.npz'
    
    if not overwrite and viz_path.exists() and results_path.exists():
        print("Test già completato, caricamento risultati esistenti...")
        try:
            # Carica i risultati esistenti
            loaded_data = np.load(str(results_path), allow_pickle=True)
            
            # Estrai i dati
            fidelity_metrics = loaded_data['fidelity_metrics'].item()
            
            print(f"RIASSUNTO (caricato):")
            print(f"  - MSE medio: {fidelity_metrics['reconstruction_quality']['avg_mse']:.6f}")
            print(f"  - SSIM medio: {fidelity_metrics['reconstruction_quality']['avg_ssim']:.4f}")
            print(f"  - Consistenza classi: {fidelity_metrics['avg_class_consistency']:.4f}")
            print(f"  - Cluster utilizzati: {fidelity_metrics['cluster_analysis']['unique_clusters_used']}/{n_clusters}")
            print(f"  - Tempo medio: {fidelity_metrics['performance']['avg_processing_time']:.3f}s")
            
            return {
                'k_neighbors': k_neighbors,
                'n_clusters': n_clusters,
                'fidelity_metrics': fidelity_metrics,
                'success': True,
                'error': None,
                'skipped': True
            }
            
        except Exception as e:
            print(f"Errore nel caricamento risultati esistenti: {e}")
            print("Rieseguo il test...")
    
    try:
        # 1. Test con un campione per ogni cifra (0-9)
        print("1. Raccolta campioni per ogni cifra...")
        data, labels = get_mnist_single_batch(max_samples=5000, split="test")
        
        # Trova un campione per ogni cifra
        digit_samples = {}
        digit_indices = {}
        
        for i, label in enumerate(labels):
            digit = label.item()
            if digit not in digit_samples:
                digit_samples[digit] = data[i]
                digit_indices[digit] = i
                
            # Fermati quando hai tutte le cifre
            if len(digit_samples) == 10:
                break
        
        print(f"   Trovati campioni per {len(digit_samples)} cifre")
        
        # Esegui quantizzazione per ogni cifra
        digit_results = []
        for digit in sorted(digit_samples.keys()):
            sample = digit_samples[digit]
            print(f"   Quantizzazione cifra {digit}...")
            
            try:
                result = quantizer.full_quantization_pipeline(sample, k_neighbors, n_clusters)
                result['true_label'] = digit
                digit_results.append(result)
                print(f"     Cluster assegnato: {result['cluster_id']}")
            except Exception as e:
                print(f"     Errore: {e}")
                continue
        
        # 2. Crea visualizzazione
        print("2. Creazione visualizzazione...")
        visualize_quantization_results(digit_results, str(viz_path))
        
        # 3. Calcola metriche di fedeltà
        print("3. Calcolo metriche di fedeltà...")
        fidelity_metrics = calculate_fidelity_metrics(quantizer, k_neighbors, n_clusters, 128)
        
        # 4. Salva risultati completi
        np.savez_compressed(
            str(results_path),
            digit_results=[r for r in digit_results],
            fidelity_metrics=fidelity_metrics,
            test_timestamp=time.time()
        )
        
        print(f"4. Risultati salvati in: {results_path}")
        
        # Mostra riassunto
        print(f"\nRIASSUNTO:")
        print(f"  - Cifre testate: {len(digit_results)}/10")
        print(f"  - MSE medio: {fidelity_metrics['reconstruction_quality']['avg_mse']:.6f}")
        print(f"  - SSIM medio: {fidelity_metrics['reconstruction_quality']['avg_ssim']:.4f}")
        print(f"  - Consistenza classi: {fidelity_metrics['avg_class_consistency']:.4f}")
        print(f"  - Cluster utilizzati: {fidelity_metrics['cluster_analysis']['unique_clusters_used']}/{n_clusters}")
        print(f"  - Tempo medio: {fidelity_metrics['performance']['avg_processing_time']:.3f}s")
        
        return {
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'digit_results': digit_results,
            'fidelity_metrics': fidelity_metrics,
            'success': True,
            'error': None,
            'skipped': False
        }
        
    except Exception as e:
        print(f"ERRORE nel test del codebook: {e}")
        return {
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'success': False,
            'error': str(e),
            'skipped': False
        }


def test_all_codebooks(overwrite: bool = False):
    """
    Testa tutti i codebook disponibili con analisi complete.
    
    Args:
        overwrite: Se True, sovrascrive tutti i test esistenti
    """
    print("TEST COMPLETO DI TUTTI I CODEBOOK")
    print("="*80)
    
    if overwrite:
        print("⚠️  MODALITÀ OVERWRITE: Tutti i test esistenti verranno sovrascritti")
    else:
        print("📋 MODALITÀ INCREMENTALE: Salta i test già completati")
    
    print("="*80)
    
    # Inizializza il quantizzatore
    quantizer = GeodeticQuantizer()
    
    # Trova tutti i codebook disponibili
    all_codebooks = []
    
    if not quantizer.codebooks_base_dir.exists():
        print(f"Directory codebooks non trovata: {quantizer.codebooks_base_dir}")
        return
    
    # Esplora tutte le cartelle k_*
    for k_folder in sorted(quantizer.codebooks_base_dir.glob('k_*')):
        if not k_folder.is_dir():
            continue
            
        k_value = int(k_folder.name.split('_')[1])
        
        # Trova tutti i file codebook in questa cartella
        for codebook_file in sorted(k_folder.glob('codebook_*.npz')):
            filename = codebook_file.name
            parts = filename.replace('.npz', '').split('_')
            
            try:
                clusters_idx = parts.index('clusters')
                n_clusters = int(parts[clusters_idx + 1])
                
                all_codebooks.append({
                    'k_neighbors': k_value,
                    'n_clusters': n_clusters,
                    'file_path': str(codebook_file)
                })
            except (ValueError, IndexError):
                print(f"Nome file non riconosciuto: {filename}")
                continue
    
    if not all_codebooks:
        print("Nessun codebook trovato!")
        return
    
    print(f"Trovati {len(all_codebooks)} codebook da testare")
    
    # Testa ogni codebook
    all_test_results = []
    successful_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for i, codebook_info in enumerate(all_codebooks):
        k_neighbors = codebook_info['k_neighbors']
        n_clusters = codebook_info['n_clusters']
        
        print(f"\nProgresso: {i+1}/{len(all_codebooks)}")
        
        try:
            test_result = test_single_codebook(quantizer, k_neighbors, n_clusters, overwrite)
            all_test_results.append(test_result)
            
            if test_result['success']:
                successful_tests += 1
                if test_result.get('skipped', False):
                    skipped_tests += 1
            else:
                failed_tests += 1
                
        except Exception as e:
            print(f"Errore critico nel test k={k_neighbors}, clusters={n_clusters}: {e}")
            failed_tests += 1
            continue
    
    # Crea report riassuntivo
    create_comprehensive_report(all_test_results, quantizer.codebooks_base_dir)
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETATI")
    print(f"{'='*80}")
    print(f"  - Codebook totali: {len(all_codebooks)}")
    print(f"  - Test riusciti: {successful_tests}")
    print(f"  - Test saltati (già esistenti): {skipped_tests}")
    print(f"  - Test nuovi eseguiti: {successful_tests - skipped_tests}")
    print(f"  - Test falliti: {failed_tests}")
    print(f"  - Report aggiornato in: {quantizer.codebooks_base_dir}")


def create_comprehensive_report(all_results: List[Dict], output_dir: Path):
    """
    Crea un report completo di tutti i test dei codebook.
    
    Args:
        all_results: Lista di tutti i risultati dei test
        output_dir: Directory dove salvare il report
    """
    print("\nCreazione report completo...")
    
    # Filtra solo i risultati di successo
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if not successful_results:
        print("Nessun test riuscito per il report!")
        return
    
    # Conta risultati saltati e nuovi
    skipped_count = sum(1 for r in successful_results if r.get('skipped', False))
    new_count = len(successful_results) - skipped_count
    
    # Crea report testuale
    report_path = output_dir / 'comprehensive_test_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("REPORT COMPLETO TEST CODEBOOK\n")
        f.write("="*50 + "\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Codebook testati con successo: {len(successful_results)}\n")
        f.write(f"  - Test nuovi eseguiti: {new_count}\n")
        f.write(f"  - Test saltati (esistenti): {skipped_count}\n\n")
        
        f.write("RIASSUNTO PER CONFIGURAZIONE:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'k-neighbors':<12} {'n_clusters':<12} {'MSE':<12} {'SSIM':<8} {'Consistency':<12} {'Clusters_Used':<15} {'Avg_Time':<10} {'Status':<10}\n")
        f.write("-" * 100 + "\n")
        
        for result in successful_results:
            if 'fidelity_metrics' not in result:
                continue
                
            fm = result['fidelity_metrics']
            k_neighbors = result['k_neighbors']
            n_clusters = result['n_clusters']
            status = "LOADED" if result.get('skipped', False) else "NEW"
            
            f.write(f"{k_neighbors:<12} "
                   f"{n_clusters:<12} "
                   f"{fm['reconstruction_quality']['avg_mse']:<12.6f} "
                   f"{fm['reconstruction_quality']['avg_ssim']:<8.4f} "
                   f"{fm['avg_class_consistency']:<12.4f} "
                   f"{fm['cluster_analysis']['unique_clusters_used']}/{n_clusters:<10} "
                   f"{fm['performance']['avg_processing_time']:<10.3f} "
                   f"{status:<10}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("ANALISI DETTAGLIATA PER CLASSE:\n")
        f.write("="*50 + "\n")
        
        for result in successful_results:
            if 'fidelity_metrics' not in result:
                continue
                
            k_neighbors = result['k_neighbors']
            n_clusters = result['n_clusters']
            fm = result['fidelity_metrics']
            
            f.write(f"\nCODEBOOK k={k_neighbors}, clusters={n_clusters}:\n")
            f.write("-" * 40 + "\n")
            
            # Verifica se ci sono dati per classe
            if 'class_consistency' in fm and fm['class_consistency']:
                for digit in range(10):
                    if str(digit) in fm['class_consistency']:
                        cc = fm['class_consistency'][str(digit)]
                        f.write(f"  Cifra {digit}: Consistency={cc['consistency']:.3f}, "
                               f"Cluster principale={cc['most_common_cluster']}, "
                               f"Campioni={cc['total_samples']}, "
                               f"Cluster unici={cc['unique_clusters']}\n")
            else:
                f.write("  Dati per classe non disponibili\n")
    
    print(f"Report testuale salvato: {report_path}")
    print(f"  - Test nuovi: {new_count}")
    print(f"  - Test caricati da file: {skipped_count}")
    
    # Crea visualizzazione riassuntiva
    create_summary_visualization(successful_results, output_dir)


def create_summary_visualization(successful_results: List[Dict], output_dir: Path):
    """
    Crea visualizzazioni riassuntive delle performance.
    
    Args:
        successful_results: Lista dei risultati di successo
        output_dir: Directory dove salvare le visualizzazioni
    """
    if not successful_results:
        return
    
    # Estrai dati per la visualizzazione
    k_neighbors_list = []
    n_clusters_list = []
    mse_list = []
    ssim_list = []
    consistency_list = []
    utilization_list = []
    
    for result in successful_results:
        if 'fidelity_metrics' not in result:
            continue
            
        fm = result['fidelity_metrics']
        k_neighbors_list.append(result['k_neighbors'])
        n_clusters_list.append(result['n_clusters'])
        mse_list.append(fm['reconstruction_quality']['avg_mse'])
        ssim_list.append(fm['reconstruction_quality']['avg_ssim'])
        consistency_list.append(fm['avg_class_consistency'])
        utilization_list.append(fm['cluster_analysis']['cluster_utilization'])
    
    # Crea plot riassuntivo
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: MSE vs n_clusters colorato per k_neighbors
    scatter1 = axes[0, 0].scatter(n_clusters_list, mse_list, c=k_neighbors_list, 
                                cmap='viridis', alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Numero di Cluster')
    axes[0, 0].set_ylabel('MSE Medio')
    axes[0, 0].set_title('MSE vs Numero di Cluster')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='k-neighbors')
    
    # Plot 2: SSIM vs n_clusters
    scatter2 = axes[0, 1].scatter(n_clusters_list, ssim_list, c=k_neighbors_list, 
                                cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Numero di Cluster')
    axes[0, 1].set_ylabel('SSIM Medio')
    axes[0, 1].set_title('SSIM vs Numero di Cluster')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='k-neighbors')
    
    # Plot 3: Consistenza delle classi
    scatter3 = axes[1, 0].scatter(n_clusters_list, consistency_list, c=k_neighbors_list, 
                                cmap='viridis', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Numero di Cluster')
    axes[1, 0].set_ylabel('Consistenza Classi')
    axes[1, 0].set_title('Consistenza Classi vs Numero di Cluster')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='k-neighbors')
    
    # Plot 4: Utilizzo cluster
    scatter4 = axes[1, 1].scatter(n_clusters_list, utilization_list, c=k_neighbors_list, 
                                cmap='viridis', alpha=0.7, s=60)
    axes[1, 1].set_xlabel('Numero di Cluster')
    axes[1, 1].set_ylabel('Utilizzo Cluster (%)')
    axes[1, 1].set_title('Utilizzo Cluster vs Numero di Cluster')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='k-neighbors')
    
    plt.tight_layout()
    
    summary_plot_path = output_dir / 'performance_summary.png'
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizzazione riassuntiva salvata: {summary_plot_path}")


def main():
    """
    Funzione principale per testare tutti i codebook.
    """
    # Gestione argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Test dei codebook per quantizzazione geodesica')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Sovrascrive tutti i test esistenti')
    
    args = parser.parse_args()
    
    print("TESTING COMPLETO QUANTIZZAZIONE GEODESICA")
    print("="*80)
    
    if args.overwrite:
        print("🔄 MODALITÀ OVERWRITE ATTIVATA")
        print("   Tutti i test esistenti verranno rigenerati")
    else:
        print("⚡ MODALITÀ INCREMENTALE ATTIVATA")
        print("   Verranno saltati i test già completati")
    
    print("="*80)
    
    # Test completo di tutti i codebook
    test_all_codebooks(overwrite=args.overwrite)
    
    print(f"\n{'='*80}")
    print("Tutti i test completati!")
    
    if not args.overwrite:
        print("\n💡 Suggerimento: Usa --overwrite per rigenerare tutti i test")


if __name__ == "__main__":
    main()
