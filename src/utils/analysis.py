import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import seaborn as sns


def get_latent_representations(model, data_loader, device, num_samples=5000):
    """
    Estrae le rappresentazioni latenti dal modello VAE.
    
    Args:
        model: Modello VAE addestrato
        data_loader: DataLoader da cui estrarre i dati
        device: Device su cui eseguire l'inferenza
        num_samples (int): Numero massimo di campioni da processare
        
    Returns:
        tuple: (latents, labels) - rappresentazioni latenti e relative etichette
    """
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in data_loader:
            if len(latents) * data.size(0) >= num_samples:
                break
            
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            # Usiamo la media (mu) come rappresentazione latente
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    return latents, labels


def visualize_pca(latents, labels, n_components=2):
    """
    Visualizza le rappresentazioni latenti usando PCA.
    
    Args:
        latents: Array delle rappresentazioni latenti
        labels: Array delle etichette
        n_components (int): Numero di componenti PCA
        
    Returns:
        PCA: Modello PCA fitted
    """
    pca = PCA(n_components=n_components)
    latents_pca = pca.fit_transform(latents)
    
    plt.figure(figsize=(12, 5))
    
    # Plot PCA
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'PCA del spazio latente VAE\nVarianza spiegata: {pca.explained_variance_ratio_.sum():.3f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    
    # Plot varianza spiegata
    plt.subplot(1, 2, 2)
    pca_full = PCA()
    pca_full.fit(latents)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Cumulativa Spiegata')
    plt.title('Varianza Spiegata PCA')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pca


def visualize_umap(latents, labels, n_neighbors=15, min_dist=0.1):
    """
    Visualizza le rappresentazioni latenti usando UMAP.
    
    Args:
        latents: Array delle rappresentazioni latenti
        labels: Array delle etichette
        n_neighbors (int): Parametro n_neighbors di UMAP
        min_dist (float): Parametro min_dist di UMAP
        
    Returns:
        umap.UMAP: Modello UMAP fitted
    """
    # Inizializzazione UMAP
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    
    latents_umap = umap_model.fit_transform(latents)
    
    plt.figure(figsize=(15, 5))
    
    # Plot UMAP colorato per digit
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(latents_umap[:, 0], latents_umap[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('UMAP del spazio latente VAE')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Plot UMAP con densità
    plt.subplot(1, 3, 2)
    plt.hexbin(latents_umap[:, 0], latents_umap[:, 1], gridsize=30, cmap='Blues')
    plt.colorbar()
    plt.title('Densità dei punti UMAP')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Plot per ogni digit separatamente
    plt.subplot(1, 3, 3)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = labels == digit
        plt.scatter(latents_umap[mask, 0], latents_umap[mask, 1], 
                   c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('UMAP per ogni digit')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.show()
    
    return umap_model


def compare_pca_umap(latents, labels):
    """
    Confronta le visualizzazioni PCA e UMAP side-by-side.
    
    Args:
        latents: Array delle rappresentazioni latenti
        labels: Array delle etichette
    """
    # PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    latents_umap = umap_model.fit_transform(latents)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot PCA
    scatter1 = axes[0].scatter(latents_pca[:, 0], latents_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[0].set_title(f'PCA\nVarianza spiegata: {pca.explained_variance_ratio_.sum():.3f}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot UMAP
    scatter2 = axes[1].scatter(latents_umap[:, 0], latents_umap[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[1].set_title('UMAP')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def analyze_umap_parameters(latents, labels):
    """
    Analizza l'effetto di diversi parametri UMAP.
    
    Args:
        latents: Array delle rappresentazioni latenti
        labels: Array delle etichette
    """
    print("Confronto con diversi parametri UMAP...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    params = [
        {'n_neighbors': 5, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.1},
        {'n_neighbors': 50, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.5}
    ]
    
    for i, param in enumerate(params):
        row, col = i // 2, i % 2
        umap_temp = umap.UMAP(**param, n_components=2, random_state=42)
        latents_temp = umap_temp.fit_transform(latents)
        
        scatter = axes[row, col].scatter(latents_temp[:, 0], latents_temp[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[row, col].set_title(f"UMAP: n_neighbors={param['n_neighbors']}, min_dist={param['min_dist']}")
        plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()