import numpy as np
import os
from pathlib import Path
import time
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore')


def load_clustering_data(clustering_file_path: str) -> Optional[Dict]:
    """
    Carica i dati di clustering da un file .npz.
    
    Args:
        clustering_file_path: Percorso al file di clustering
        
    Returns:
        Dizionario con i dati di clustering o None se errore
    """
    try:
        data = np.load(clustering_file_path, allow_pickle=True)
        
        # Verifica che contengano le chiavi necessarie
        required_keys = ['latent_vectors', 'geodesic_centers', 'config']
        for key in required_keys:
            if key not in data:
                print(f"Chiave mancante '{key}' in {clustering_file_path}")
                return None
        
        return {
            'latent_vectors': data['latent_vectors'],
            'geodesic_centers': data['geodesic_centers'],
            'geodesic_labels': data['geodesic_labels'],
            'true_labels': data['true_labels'],
            'config': data['config'].item()  # Convert from numpy array to dict
        }
        
    except Exception as e:
        print(f"Errore nel caricamento di {clustering_file_path}: {e}")
        return None


def create_codebook_from_centers(latent_vectors: np.ndarray, center_indices: np.ndarray) -> np.ndarray:
    """
    Crea un codebook (matrice KxD) dai centroidi dei cluster.
    
    Args:
        latent_vectors: Vettori latenti originali [N, D]
        center_indices: Indici dei centroidi [K,]
        
    Returns:
        Codebook matrice [K, D] dove ogni riga è un vettore centroide
    """
    # I centroidi sono rappresentati come indici nei vettori latenti originali
    codebook = latent_vectors[center_indices]
    return codebook


def save_codebook(
    codebook: np.ndarray,
    config: Dict,
    original_clustering_data: Dict,
    output_path: str
):
    """
    Salva il codebook e i metadati associati.
    
    Args:
        codebook: Matrice codebook [K, D]
        config: Configurazione originale del clustering
        original_clustering_data: Dati originali del clustering
        output_path: Percorso di output
    """
    # Metadati del codebook
    codebook_info = {
        'codebook_size': codebook.shape[0],  # K (numero di cluster)
        'vector_dim': codebook.shape[1],     # D (dimensione vettore latente)
        'source_clustering_config': config,
        'creation_timestamp': time.time()
    }
    
    # Salva il codebook e i metadati
    np.savez_compressed(
        output_path,
        codebook=codebook,
        codebook_info=codebook_info,
        # Mantieni alcuni dati originali per riferimento
        original_config=config,
        geodesic_centers_indices=original_clustering_data['geodesic_centers'],
        geodesic_labels=original_clustering_data['geodesic_labels']
    )
    
    print(f"Codebook salvato: {output_path}")
    print(f"  - Dimensioni: {codebook.shape[0]} clusters x {codebook.shape[1]} features")


def process_clustering_folder(clustering_folder_path: str, codebook_folder_path: str) -> Dict:
    """
    Processa tutti i file di clustering in una cartella e crea i codebook corrispondenti.
    
    Args:
        clustering_folder_path: Percorso alla cartella dei clustering
        codebook_folder_path: Percorso alla cartella di output per i codebook
        
    Returns:
        Dizionario con statistiche del processamento
    """
    clustering_path = Path(clustering_folder_path)
    codebook_path = Path(codebook_folder_path)
    
    # Crea la directory di output se non esiste
    codebook_path.mkdir(parents=True, exist_ok=True)
    
    # Trova tutti i file di clustering (esclude i grafici e le immagini)
    clustering_files = list(clustering_path.glob('clustering_k_*_clusters_*.npz'))
    
    if not clustering_files:
        print(f"Nessun file di clustering trovato in {clustering_folder_path}")
        return {'processed': 0, 'errors': 0}
    
    print(f"\nProcessamento cartella: {clustering_folder_path}")
    print(f"File di clustering trovati: {len(clustering_files)}")
    
    stats = {'processed': 0, 'errors': 0, 'codebooks_created': []}
    
    for clustering_file in clustering_files:
        try:
            print(f"\nProcessamento: {clustering_file.name}")
            
            # Carica i dati di clustering
            clustering_data = load_clustering_data(str(clustering_file))
            if clustering_data is None:
                stats['errors'] += 1
                continue
            
            # Estrai i componenti necessari
            latent_vectors = clustering_data['latent_vectors']
            center_indices = clustering_data['geodesic_centers']
            config = clustering_data['config']
            
            # Crea il codebook
            codebook = create_codebook_from_centers(latent_vectors, center_indices)
            
            # Genera il nome del file di output
            output_filename = clustering_file.name.replace('clustering_', 'codebook_')
            output_path = codebook_path / output_filename
            
            # Salva il codebook
            save_codebook(codebook, config, clustering_data, str(output_path))
            
            stats['processed'] += 1
            stats['codebooks_created'].append({
                'file': str(output_path),
                'shape': codebook.shape,
                'k_neighbors': config['k_neighbors'],
                'n_clusters': config['n_clusters']
            })
            
        except Exception as e:
            print(f"Errore nel processamento di {clustering_file.name}: {e}")
            stats['errors'] += 1
            continue
    
    return stats


def create_summary_report(all_stats: List[Dict], output_dir: str):
    """
    Crea un report riassuntivo di tutti i codebook creati.
    
    Args:
        all_stats: Lista delle statistiche per ogni cartella processata
        output_dir: Directory di output principale
    """
    print(f"\n{'='*80}")
    print("REPORT RIASSUNTIVO CREAZIONE CODEBOOK")
    print(f"{'='*80}")
    
    total_processed = sum(stat['processed'] for stat in all_stats)
    total_errors = sum(stat['errors'] for stat in all_stats)
    
    print(f"Totale codebook creati: {total_processed}")
    print(f"Totale errori: {total_errors}")
    
    # Crea tabella riassuntiva
    print(f"\nDETTAGLI PER CONFIGURAZIONE:")
    print(f"{'k-neighbors':<12} {'n_clusters':<12} {'Codebook Shape':<20} {'File':<50}")
    print("-" * 100)
    
    for stat in all_stats:
        for codebook_info in stat['codebooks_created']:
            print(f"{codebook_info['k_neighbors']:<12} "
                  f"{codebook_info['n_clusters']:<12} "
                  f"{str(codebook_info['shape']):<20} "
                  f"{Path(codebook_info['file']).name:<50}")
    
    # Salva report su file
    report_path = Path(output_dir) / 'codebook_creation_report.txt'
    with open(report_path, 'w') as f:
        f.write("REPORT CREAZIONE CODEBOOK\n")
        f.write("="*50 + "\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Totale codebook creati: {total_processed}\n")
        f.write(f"Totale errori: {total_errors}\n\n")
        
        f.write("DETTAGLI:\n")
        for stat in all_stats:
            for codebook_info in stat['codebooks_created']:
                f.write(f"k={codebook_info['k_neighbors']}, "
                        f"clusters={codebook_info['n_clusters']}, "
                        f"shape={codebook_info['shape']}, "
                        f"file={codebook_info['file']}\n")
    
    print(f"\nReport salvato: {report_path}")


def main():
    """
    Funzione principale che processa tutti i clustering e crea i codebook.
    """
    print("AVVIO CREAZIONE CODEBOOK DAI CLUSTERING GEODESICI")
    print("="*80)
    
    # Percorsi base
    clustering_base_dir = 'output/clustering_knn_graphs'
    codebook_base_dir = 'output/codebooks'
    
    # Verifica che la directory di clustering esista
    if not os.path.exists(clustering_base_dir):
        print(f"Directory di clustering non trovata: {clustering_base_dir}")
        return
    
    # Trova tutte le cartelle k_*
    clustering_base_path = Path(clustering_base_dir)
    k_folders = [f for f in clustering_base_path.iterdir() 
                 if f.is_dir() and f.name.startswith('k_')]
    
    if not k_folders:
        print(f"Nessuna cartella k_* trovata in {clustering_base_dir}")
        return
    
    print(f"Cartelle k-neighbors trovate: {len(k_folders)}")
    for folder in sorted(k_folders):
        print(f"  - {folder.name}")
    
    # Processa ogni cartella
    all_stats = []
    start_time = time.time()
    
    for k_folder in sorted(k_folders):
        k_value = k_folder.name  # es. "k_10"
        
        # Percorsi di input e output
        clustering_folder_path = str(k_folder)
        codebook_folder_path = os.path.join(codebook_base_dir, k_value)
        
        # Processa la cartella
        stats = process_clustering_folder(clustering_folder_path, codebook_folder_path)
        all_stats.append(stats)
        
        print(f"\nRisultati per {k_value}:")
        print(f"  - Codebook creati: {stats['processed']}")
        print(f"  - Errori: {stats['errors']}")
    
    # Tempo totale
    elapsed_time = time.time() - start_time
    
    # Crea report finale
    create_summary_report(all_stats, codebook_base_dir)
    
    print(f"\nProcessamento completato in {elapsed_time:.1f} secondi")
    print(f"Codebook salvati in: {codebook_base_dir}")


if __name__ == "__main__":
    main()
