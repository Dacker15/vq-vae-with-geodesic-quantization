import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Aggiungi il percorso src al path per importare i moduli
try:
    sys.path.append(str(Path(__file__).parent))
except NameError:
    # Caso quando __file__ non è definito (exec, ecc.)
    sys.path.append('src')

from models.transformer import create_transformer_model
from models.vae import VAE
from utils.device import device
from utils.dataloader import get_mnist_single_batch


def load_transformer_model(k_neighbors: int, n_clusters: int) -> Dict:
    """
    Carica un modello Transformer addestrato.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        
    Returns:
        Dizionario con modello e metadati
    """
    model_path = f'src/output/transformer/k_{k_neighbors}/transformer_k_{k_neighbors}_clusters_{n_clusters}.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello Transformer non trovato: {model_path}")
      # Carica checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Ricrea il modello
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    
    model = create_transformer_model(
        input_dim=input_dim,
        output_dim=output_dim,
        model_type='simple',
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    )
    
    # Carica i pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'scaler_encoded': checkpoint['scaler_encoded'],
        'scaler_quantized': checkpoint['scaler_quantized'],
        'config': checkpoint['config'],
        'input_dim': input_dim,
        'output_dim': output_dim
    }


def load_vae_model(latent_dim: int) -> VAE:
    """
    Carica un modello VAE.
    
    Args:
        latent_dim: Dimensione dello spazio latente
        
    Returns:
        Modello VAE caricato
    """
    vae_path = f'src/output/vae/vae_model_{latent_dim}d.pth'
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Modello VAE non trovato: {vae_path}")
    
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def get_latent_dim_for_clustering(k_neighbors: int, n_clusters: int) -> int:
    """
    Ottiene la dimensione latente per una specifica configurazione di clustering.
    """
    clustering_path = f'src/output/clustering_knn_graphs/k_{k_neighbors}/clustering_k_{k_neighbors}_clusters_{n_clusters}.npz'
    
    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"File clustering non trovato: {clustering_path}")
    
    clustering_data = np.load(clustering_path, allow_pickle=True)
    config = clustering_data['config'].item()
    return config['latent_dim']


def test_transformer_on_samples(k_neighbors: int, n_clusters: int, num_test_samples: int = 100) -> Dict:
    """
    Testa un modello Transformer su campioni del test set.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        num_test_samples: Numero di campioni di test
        
    Returns:
        Dizionario con risultati del test
    """
    print(f"\nTest Transformer per k={k_neighbors}, clusters={n_clusters}")
    
    try:
        # Carica modello Transformer
        transformer_data = load_transformer_model(k_neighbors, n_clusters)
        transformer_model = transformer_data['model']
        scaler_encoded = transformer_data['scaler_encoded']
        scaler_quantized = transformer_data['scaler_quantized']
        
        # Ottieni dimensione latente e carica VAE
        latent_dim = get_latent_dim_for_clustering(k_neighbors, n_clusters)
        vae_model = load_vae_model(latent_dim)
          # Carica codebook per confronto
        codebook_path = f'src/output/codebooks/k_{k_neighbors}/codebook_k_{k_neighbors}_clusters_{n_clusters}.npz'
        codebook_data = np.load(codebook_path, allow_pickle=True)
        codebook = codebook_data['codebook']
        
        # Ottieni campioni di test MNIST
        test_data, test_labels = get_mnist_single_batch(max_samples=num_test_samples, split="test")
        
        # Lista per salvare risultati
        results = []
        predictions = []
        ground_truth = []
        
        print(f"  Testing su {len(test_data)} campioni...")
        
        with torch.no_grad():
            for i, (sample, label) in enumerate(zip(test_data, test_labels)):
                # 1. Encoding con VAE
                sample = sample.to(device).view(1, -1)  # [1, 784]
                mu, logvar = vae_model.encode(sample)
                latent_vector = mu.cpu().numpy()  # [1, latent_dim]
                
                # 2. Normalizzazione per Transformer
                latent_normalized = scaler_encoded.transform(latent_vector)
                
                # 3. Predizione con Transformer
                latent_tensor = torch.FloatTensor(latent_normalized).to(device)
                predicted_centroid_normalized = transformer_model(latent_tensor)
                
                # 4. Denormalizzazione
                predicted_centroid = scaler_quantized.inverse_transform(
                    predicted_centroid_normalized.cpu().numpy()
                )
                
                # 5. Trova il centroide più vicino nel codebook (per confronto)
                distances = np.linalg.norm(codebook - predicted_centroid, axis=1)
                closest_cluster_id = np.argmin(distances)
                actual_centroid = codebook[closest_cluster_id]
                
                # 6. Ricostruzione con VAE
                predicted_centroid_tensor = torch.FloatTensor(predicted_centroid).to(device)
                reconstructed_predicted = vae_model.decode(predicted_centroid_tensor)
                
                actual_centroid_tensor = torch.FloatTensor(actual_centroid.reshape(1, -1)).to(device)
                reconstructed_actual = vae_model.decode(actual_centroid_tensor)
                
                # 7. Ricostruzione diretta (per confronto)
                reconstructed_direct = vae_model.decode(mu)
                
                # Salva risultati
                result = {
                    'sample_idx': i,
                    'true_label': label.item(),
                    'original_image': sample.cpu().numpy().reshape(28, 28),
                    'latent_vector': latent_vector.flatten(),
                    'predicted_centroid': predicted_centroid.flatten(),
                    'actual_centroid': actual_centroid,
                    'closest_cluster_id': closest_cluster_id,
                    'reconstructed_predicted': reconstructed_predicted.cpu().numpy().reshape(28, 28),
                    'reconstructed_actual': reconstructed_actual.cpu().numpy().reshape(28, 28),
                    'reconstructed_direct': reconstructed_direct.cpu().numpy().reshape(28, 28),
                    'prediction_error': np.linalg.norm(predicted_centroid - actual_centroid)
                }
                
                results.append(result)
                predictions.append(predicted_centroid.flatten())
                ground_truth.append(actual_centroid)
                
                if (i + 1) % 20 == 0:
                    print(f"    Processati {i + 1}/{len(test_data)} campioni")
        
        # Calcola metriche globali
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        
        # Errori per singola dimensione
        dimension_errors = np.mean(np.abs(predictions - ground_truth), axis=0)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'dimension_errors': dimension_errors,
            'mean_prediction_error': np.mean([r['prediction_error'] for r in results])
        }
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  Errore medio predizione: {metrics['mean_prediction_error']:.6f}")
        
        return {
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'results': results,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"  Errore nel test: {e}")
        return {
            'k_neighbors': k_neighbors,
            'n_clusters': n_clusters,
            'error': str(e),
            'success': False
        }


def visualize_test_results(test_results: Dict, num_samples_to_show: int = 10):
    """
    Visualizza i risultati del test del Transformer.
    
    Args:
        test_results: Risultati del test
        num_samples_to_show: Numero di campioni da mostrare
    """
    if not test_results['success']:
        print(f"Test fallito: {test_results.get('error', 'Errore sconosciuto')}")
        return
    
    k_neighbors = test_results['k_neighbors']
    n_clusters = test_results['n_clusters']
    results = test_results['results']
    
    # Seleziona campioni da mostrare
    samples_to_show = results[:num_samples_to_show]
    
    # Crea visualizzazione
    fig, axes = plt.subplots(4, len(samples_to_show), figsize=(2*len(samples_to_show), 8))
    
    if len(samples_to_show) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, result in enumerate(samples_to_show):
        # Riga 1: Immagine originale
        axes[0, i].imshow(result['original_image'], cmap='gray')
        axes[0, i].set_title(f'Orig: {result["true_label"]}')
        axes[0, i].axis('off')
        
        # Riga 2: Ricostruzione diretta VAE
        axes[1, i].imshow(result['reconstructed_direct'], cmap='gray')
        axes[1, i].set_title('VAE Direct')
        axes[1, i].axis('off')
        
        # Riga 3: Ricostruzione con predizione Transformer
        axes[2, i].imshow(result['reconstructed_predicted'], cmap='gray')
        axes[2, i].set_title(f'Transformer\nErr: {result["prediction_error"]:.3f}')
        axes[2, i].axis('off')
        
        # Riga 4: Ricostruzione con centroide reale più vicino
        axes[3, i].imshow(result['reconstructed_actual'], cmap='gray')
        axes[3, i].set_title(f'True Centroid\nCluster: {result["closest_cluster_id"]}')
        axes[3, i].axis('off')
    
    # Etichette delle righe
    fig.text(0.02, 0.875, 'Originale', rotation=90, va='center', fontweight='bold')
    fig.text(0.02, 0.625, 'VAE\nDiretto', rotation=90, va='center', fontweight='bold')
    fig.text(0.02, 0.375, 'Transformer\nPredizione', rotation=90, va='center', fontweight='bold')
    fig.text(0.02, 0.125, 'Centroide\nReale', rotation=90, va='center', fontweight='bold')
    
    plt.suptitle(f'Test Transformer - k={k_neighbors}, clusters={n_clusters}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
      # Output layers
    output_dir = f'src/output/transformer/k_{k_neighbors}'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = f'{output_dir}/test_results_k_{k_neighbors}_clusters_{n_clusters}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualizzazione salvata: {plot_path}")
    
    return plot_path


def create_metrics_summary(all_test_results: List[Dict], output_dir: str = 'src/output/transformer'):
    """
    Crea un riassunto delle metriche per tutti i test.
    """
    # Filtra solo i risultati di successo
    successful_results = [r for r in all_test_results if r['success']]
    
    if not successful_results:
        print("Nessun test completato con successo")
        return
    
    # Estrai dati per il riassunto
    k_values = []
    n_clusters_values = []
    mse_values = []
    mae_values = []
    rmse_values = []
    
    for result in successful_results:
        k_values.append(result['k_neighbors'])
        n_clusters_values.append(result['n_clusters'])
        mse_values.append(result['metrics']['mse'])
        mae_values.append(result['metrics']['mae'])
        rmse_values.append(result['metrics']['rmse'])
    
    # Crea DataFrame
    df = pd.DataFrame({
        'k_neighbors': k_values,
        'n_clusters': n_clusters_values,
        'mse': mse_values,
        'mae': mae_values,
        'rmse': rmse_values
    })
    
    # Salva CSV
    csv_path = f'{output_dir}/test_metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    
    # Crea visualizzazione delle metriche
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: MSE vs n_clusters
    scatter1 = axes[0, 0].scatter(n_clusters_values, mse_values, c=k_values, 
                                  cmap='viridis', alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Numero di Cluster')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE vs Numero di Cluster')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='k-neighbors')
    
    # Plot 2: MAE vs n_clusters
    scatter2 = axes[0, 1].scatter(n_clusters_values, mae_values, c=k_values, 
                                  cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Numero di Cluster')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE vs Numero di Cluster')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='k-neighbors')
    
    # Plot 3: RMSE vs k_neighbors
    scatter3 = axes[1, 0].scatter(k_values, rmse_values, c=n_clusters_values, 
                                  cmap='plasma', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('k-neighbors')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('RMSE vs k-neighbors')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='n_clusters')
    
    # Plot 4: Confronto MSE e MAE
    axes[1, 1].scatter(mse_values, mae_values, c=k_values, 
                       cmap='viridis', alpha=0.7, s=60)
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MSE vs MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Riassunto Metriche Test Transformer', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Salva grafico
    plot_path = f'{output_dir}/test_metrics_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Riassunto metriche salvato: {csv_path}")
    print(f"Grafico riassunto salvato: {plot_path}")
    
    # Stampa statistiche
    print(f"\nStatistiche riassuntive:")
    print(f"MSE medio: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"MAE medio: {np.mean(mae_values):.6f} ± {np.std(mae_values):.6f}")
    print(f"RMSE medio: {np.mean(rmse_values):.6f} ± {np.std(rmse_values):.6f}")


def main():
    """
    Funzione principale per il testing dei modelli Transformer.
    """
    print("AVVIO TESTING MODELLI TRANSFORMER")
    print("="*80)
    
    # Parametri di default
    k_values = [8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 25]
    n_clusters_values = [64, 96, 128, 256, 512]
    num_test_samples = 50  # Numero di campioni di test per ogni modello
      # Verifica che esistano i modelli
    transformer_base_dir = 'src/output/transformer'
    if not os.path.exists(transformer_base_dir):
        print(f"Directory transformer non trovata: {transformer_base_dir}")
        return
    
    all_test_results = []
    total_tests = 0
    successful_tests = 0
    errors = []
    
    start_time = time.time()
    
    for k in k_values:
        for n_clusters in n_clusters_values:
            total_tests += 1
            
            print(f"\n--- Test {total_tests}: k={k}, clusters={n_clusters} ---")
              # Verifica se il modello esiste
            model_path = f'src/output/transformer/k_{k}/transformer_k_{k}_clusters_{n_clusters}.pth'
            if not os.path.exists(model_path):
                print(f"  Modello non trovato: {model_path}")
                continue
            
            try:
                # Testa il modello
                test_result = test_transformer_on_samples(k, n_clusters, num_test_samples)
                all_test_results.append(test_result)
                
                if test_result['success']:
                    successful_tests += 1
                    
                    # Crea visualizzazione
                    visualize_test_results(test_result, num_samples_to_show=8)
                else:
                    errors.append(f"k={k}, clusters={n_clusters}: {test_result.get('error', 'Errore sconosciuto')}")
                    
            except Exception as e:
                error_msg = f"k={k}, clusters={n_clusters}: {str(e)}"
                errors.append(error_msg)
                print(f"  Errore: {e}")
                continue
    
    # Crea riassunto finale
    if successful_tests > 0:
        create_metrics_summary(all_test_results)
    
    # Statistiche finali
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("STATISTICHE FINALI TESTING TRANSFORMER")
    print(f"{'='*80}")
    print(f"Test totali eseguiti: {total_tests}")
    print(f"Test completati con successo: {successful_tests}")
    print(f"Errori: {len(errors)}")
    print(f"Tempo totale: {elapsed_time:.1f} secondi")
    
    if errors:
        print(f"\nErrori riscontrati:")
        for error in errors[:10]:  # Mostra solo i primi 10 errori
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10} errori")
    
    print(f"\nRisultati salvati in: {transformer_base_dir}/")


if __name__ == "__main__":
    main()
