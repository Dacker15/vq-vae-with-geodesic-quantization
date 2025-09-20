import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from sklearn.model_selection import train_test_split
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
from models.knn_graph import KNNGraph
from utils.device import device
from utils.dataloader import get_mnist_single_batch


class TransformerDataset(Dataset):
    """
    Dataset per il training del Transformer.
    """
    
    def __init__(self, encoded_vectors, quantized_vectors):
        """
        Args:
            encoded_vectors: Array numpy [N, D] - vettori latenti VAE
            quantized_vectors: Array numpy [N, D] - centroidi corrispondenti
        """
        self.encoded = torch.FloatTensor(encoded_vectors)
        self.quantized = torch.FloatTensor(quantized_vectors)
        
    def __len__(self):
        return len(self.encoded)
    
    def __getitem__(self, idx):
        return self.encoded[idx], self.quantized[idx]


def load_clustering_and_vae_data(k_neighbors: int, n_clusters: int) -> Dict:
    """
    Carica i dati di clustering e il modello VAE corrispondente.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        
    Returns:
        Dizionario con i dati caricati
    """
    print(f"Caricamento dati per k={k_neighbors}, clusters={n_clusters}")
      # Percorsi ai file
    clustering_path = f'src/output/clustering_knn_graphs/k_{k_neighbors}/clustering_k_{k_neighbors}_clusters_{n_clusters}.npz'
    codebook_path = f'src/output/codebooks/k_{k_neighbors}/codebook_k_{k_neighbors}_clusters_{n_clusters}.npz'
    
    # Verifica esistenza file
    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"File clustering non trovato: {clustering_path}")
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"File codebook non trovato: {codebook_path}")
    
    # Carica dati clustering
    clustering_data = np.load(clustering_path, allow_pickle=True)
    config = clustering_data['config'].item()
    latent_dim = config['latent_dim']
    
    # Carica codebook
    codebook_data = np.load(codebook_path, allow_pickle=True)
    codebook = codebook_data['codebook']
      # Carica modello VAE
    vae_path = f'src/output/vae/vae_model_{latent_dim}d.pth'
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Modello VAE non trovato: {vae_path}")
    
    vae_model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.to(device)
    vae_model.eval()
    
    return {
        'clustering_data': clustering_data,
        'codebook': codebook,
        'vae_model': vae_model,
        'latent_dim': latent_dim,
        'config': config
    }


def create_dataset_from_clustering(k_neighbors: int, n_clusters: int, max_samples: int = 5000) -> bool:
    """
    Crea un dataset CSV dal clustering esistente.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        max_samples: Numero massimo di campioni da utilizzare
        
    Returns:
        True se il dataset è stato creato con successo
    """
    dataset_path = f'src/datasets/dataset_{k_neighbors}_{n_clusters}.csv'
    
    # Se il file esiste già, salta la creazione
    if os.path.exists(dataset_path):
        print(f"Dataset già esistente: {dataset_path}")
        return True
    
    print(f"Creazione dataset per k={k_neighbors}, clusters={n_clusters}")
    
    try:
        # Carica i dati
        data = load_clustering_and_vae_data(k_neighbors, n_clusters)
        clustering_data = data['clustering_data']
        codebook = data['codebook']
        
        # Estrai i vettori latenti e le etichette dei cluster
        latent_vectors = clustering_data['latent_vectors']
        cluster_labels = clustering_data['geodesic_labels']
        
        # Limita il numero di campioni se necessario
        if len(latent_vectors) > max_samples:
            indices = np.random.choice(len(latent_vectors), max_samples, replace=False)
            latent_vectors = latent_vectors[indices]
            cluster_labels = cluster_labels[indices]
        
        # Crea le coppie encoded-quantized
        encoded_list = []
        quantized_list = []
        
        for i, (encoded_vec, cluster_id) in enumerate(zip(latent_vectors, cluster_labels)):
            if cluster_id < len(codebook):  # Verifica validità del cluster_id
                quantized_vec = codebook[cluster_id]
                encoded_list.append(encoded_vec)
                quantized_list.append(quantized_vec)
        
        # Converti in array numpy
        encoded_array = np.array(encoded_list)
        quantized_array = np.array(quantized_list)
        
        print(f"  Dataset creato: {len(encoded_array)} campioni")
        print(f"  Dimensione encoded: {encoded_array.shape}")
        print(f"  Dimensione quantized: {quantized_array.shape}")
        
        # Crea DataFrame e salva CSV
        # Flatten dei vettori per salvare in CSV
        encoded_cols = [f'encoded_{i}' for i in range(encoded_array.shape[1])]
        quantized_cols = [f'quantized_{i}' for i in range(quantized_array.shape[1])]
        
        # Crea DataFrame
        df_data = {}
        for i, col in enumerate(encoded_cols):
            df_data[col] = encoded_array[:, i]
        for i, col in enumerate(quantized_cols):
            df_data[col] = quantized_array[:, i]
        
        df = pd.DataFrame(df_data)
        
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Salva CSV
        df.to_csv(dataset_path, index=False)
        print(f"  Dataset salvato: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"Errore nella creazione del dataset: {e}")
        return False


def load_dataset_from_csv(k_neighbors: int, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica un dataset da file CSV.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        
    Returns:
        Tuple (encoded_vectors, quantized_vectors)
    """
    dataset_path = f'src/datasets/dataset_{k_neighbors}_{n_clusters}.csv'
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Separa le colonne encoded e quantized
    encoded_cols = [col for col in df.columns if col.startswith('encoded_')]
    quantized_cols = [col for col in df.columns if col.startswith('quantized_')]
    
    encoded_vectors = df[encoded_cols].values
    quantized_vectors = df[quantized_cols].values
    
    return encoded_vectors, quantized_vectors


def train_transformer(k_neighbors: int, n_clusters: int, epochs: int = 100, 
                     batch_size: int = 64, learning_rate: float = 1e-4) -> Dict:
    """
    Addestra un modello Transformer per predire centroidi.
    
    Args:
        k_neighbors: Valore k del grafo k-NN
        n_clusters: Numero di cluster
        epochs: Numero di epoche di training
        batch_size: Dimensione del batch
        learning_rate: Learning rate
        
    Returns:
        Dizionario con risultati del training
    """
    print(f"\nAddestramento Transformer per k={k_neighbors}, clusters={n_clusters}")
      # Percorso per salvare il modello
    model_dir = f'src/output/transformer/k_{k_neighbors}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/transformer_k_{k_neighbors}_clusters_{n_clusters}.pth'
    
    # Se il modello esiste già, salta l'addestramento
    if os.path.exists(model_path):
        print(f"Modello già esistente: {model_path}")
        return {'model_path': model_path, 'skipped': True}
    
    try:
        # Carica dataset
        encoded_vectors, quantized_vectors = load_dataset_from_csv(k_neighbors, n_clusters)
        
        # Normalizzazione
        scaler_encoded = StandardScaler()
        scaler_quantized = StandardScaler()
        
        encoded_normalized = scaler_encoded.fit_transform(encoded_vectors)
        quantized_normalized = scaler_quantized.fit_transform(quantized_vectors)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_normalized, quantized_normalized, 
            test_size=0.2, random_state=42
        )
        
        # Crea datasets e dataloaders
        train_dataset = TransformerDataset(X_train, y_train)
        test_dataset = TransformerDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Crea modello
        input_dim = encoded_vectors.shape[1]
        output_dim = quantized_vectors.shape[1]
        
        model = create_transformer_model(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type='simple',
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1
        )
        model = model.to(device)
        
        # Optimizer e loss function
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        print(f"Avvio training: {len(train_dataset)} campioni train, {len(test_dataset)} campioni test")
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (encoded, quantized) in enumerate(train_loader):
                encoded, quantized = encoded.to(device), quantized.to(device)
                
                optimizer.zero_grad()
                predicted = model(encoded)
                loss = criterion(predicted, quantized)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_test_loss = 0.0
            
            with torch.no_grad():
                for encoded, quantized in test_loader:
                    encoded, quantized = encoded.to(device), quantized.to(device)
                    predicted = model(encoded)
                    loss = criterion(predicted, quantized)
                    epoch_test_loss += loss.item()
            
            avg_test_loss = epoch_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # Scheduler step
            scheduler.step(avg_test_loss)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_encoded': scaler_encoded,
                    'scaler_quantized': scaler_quantized,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'config': {
                        'k_neighbors': k_neighbors,
                        'n_clusters': n_clusters,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate
                    }
                }, model_path)
            
            if epoch % 10 == 0:
                print(f"Epoca {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
        
        # Salva grafico delle losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - k={k_neighbors}, clusters={n_clusters}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f'{model_dir}/training_loss_k_{k_neighbors}_clusters_{n_clusters}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Modello salvato: {model_path}")
        print(f"Grafico salvato: {plot_path}")
        print(f"Best test loss: {best_test_loss:.6f}")
        
        return {
            'model_path': model_path,
            'plot_path': plot_path,
            'best_test_loss': best_test_loss,
            'final_train_loss': avg_train_loss,
            'skipped': False
        }
        
    except Exception as e:
        print(f"Errore durante il training: {e}")
        return {'error': str(e), 'skipped': False}


def main():
    """
    Funzione principale per l'addestramento dei modelli Transformer.
    """
    print("AVVIO ADDESTRAMENTO MODELLI TRANSFORMER")
    print("="*80)
    
    # Parametri di default
    k_values = [8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 25]
    n_clusters_values = [64, 96, 128, 256, 512]
      # Verifica che esistano i dati necessari
    clustering_base_dir = 'src/output/clustering_knn_graphs'
    if not os.path.exists(clustering_base_dir):
        print(f"Directory clustering non trovata: {clustering_base_dir}")
        return
    
    total_combinations = 0
    successful_datasets = 0
    successful_trainings = 0
    skipped_trainings = 0
    errors = []
    
    start_time = time.time()
    
    for k in k_values:
        for n_clusters in n_clusters_values:
            total_combinations += 1
            
            print(f"\n--- Processamento {total_combinations}: k={k}, clusters={n_clusters} ---")
            
            try:
                # 1. Crea dataset
                dataset_success = create_dataset_from_clustering(k, n_clusters)
                if dataset_success:
                    successful_datasets += 1
                    
                    # 2. Addestra modello
                    training_result = train_transformer(k, n_clusters)
                    
                    if 'error' not in training_result:
                        if training_result.get('skipped', False):
                            skipped_trainings += 1
                        else:
                            successful_trainings += 1
                    else:
                        errors.append(f"k={k}, clusters={n_clusters}: {training_result['error']}")
                else:
                    errors.append(f"k={k}, clusters={n_clusters}: Errore nella creazione del dataset")
                    
            except Exception as e:
                errors.append(f"k={k}, clusters={n_clusters}: {str(e)}")
                continue
    
    # Statistiche finali
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("STATISTICHE FINALI ADDESTRAMENTO TRANSFORMER")
    print(f"{'='*80}")
    print(f"Combinazioni totali processate: {total_combinations}")
    print(f"Dataset creati con successo: {successful_datasets}")
    print(f"Modelli addestrati con successo: {successful_trainings}")
    print(f"Modelli saltati (già esistenti): {skipped_trainings}")
    print(f"Errori: {len(errors)}")
    print(f"Tempo totale: {elapsed_time:.1f} secondi")
    
    if errors:
        print(f"\nErrori riscontrati:")
        for error in errors[:10]:  # Mostra solo i primi 10 errori
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10} errori")
    
    print("Modelli salvati in: src/output/transformer/")


if __name__ == "__main__":
    main()
