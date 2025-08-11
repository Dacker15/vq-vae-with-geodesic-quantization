import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import torch.nn.functional as F


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
        random_state=42,
        n_jobs=1
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
    umap_model = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
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
        umap_temp = umap.UMAP(**param, n_components=2, random_state=42, n_jobs=1)
        latents_temp = umap_temp.fit_transform(latents)
        
        scatter = axes[row, col].scatter(latents_temp[:, 0], latents_temp[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[row, col].set_title(f"UMAP: n_neighbors={param['n_neighbors']}, min_dist={param['min_dist']}")
        plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()


def sample_from_latent_space(model, data_loader, device):
    """
    Genera un confronto tra immagini originali e ricostruzioni del VAE per ogni cifra (0-9).
    Crea un grafico 2x10 dove:
    - Riga 1: Ricostruzioni del VAE per ogni cifra
    - Riga 2: Immagini originali corrispondenti
    
    Args:
        model: Modello VAE addestrato
        data_loader: DataLoader con dati etichettati
        device: Device su cui eseguire l'inferenza
        num_samples (int): Parametro mantenuto per compatibilità (non utilizzato)
    """
    model.eval()
    
    # Dizionario per memorizzare un esempio per ogni cifra
    examples_by_digit = {}
    reconstructions_by_digit = {}
    
    # Cerca un esempio per ogni cifra (0-9)
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            for digit in range(10):
                if digit not in examples_by_digit:
                    # Trova il primo esempio di questa cifra
                    mask = labels == digit
                    if mask.any():
                        # Prendi il primo esempio di questa cifra
                        example_idx = mask.nonzero(as_tuple=True)[0][0]
                        original_image = data[example_idx:example_idx+1]
                        
                        # Salva l'immagine originale
                        examples_by_digit[digit] = original_image.cpu().numpy().reshape(28, 28)
                        
                        # Genera la ricostruzione
                        mu, logvar = model.encode(original_image.view(-1, 784))
                        reconstructed = model.decode(mu)
                        reconstructions_by_digit[digit] = reconstructed.cpu().numpy().reshape(28, 28)
            
            # Esci dal loop quando abbiamo trovato esempi per tutte le 10 cifre
            if len(examples_by_digit) == 10:
                break
    
    # Verifica che abbiamo trovato esempi per tutte le cifre
    if len(examples_by_digit) < 10:
        missing_digits = [d for d in range(10) if d not in examples_by_digit]
        print(f"⚠️  Attenzione: Non sono stati trovati esempi per le cifre: {missing_digits}")
        return
    
    # Crea il grafico 2x10
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    
    for digit in range(10):
        # Riga 1: Ricostruzioni del VAE
        axes[0, digit].imshow(reconstructions_by_digit[digit], cmap='gray')
        axes[0, digit].set_title(f'VAE: {digit}', fontsize=12, fontweight='bold')
        axes[0, digit].axis('off')
        
        # Riga 2: Immagini originali
        axes[1, digit].imshow(examples_by_digit[digit], cmap='gray')
        axes[1, digit].set_title(f'Orig: {digit}', fontsize=12, color='darkblue')
        axes[1, digit].axis('off')
    
    # Aggiungi etichette per le righe
    fig.text(0.02, 0.75, 'Ricostruzioni\nVAE', fontsize=14, fontweight='bold', 
             ha='center', va='center', rotation=90)
    fig.text(0.02, 0.25, 'Immagini\nOriginali', fontsize=14, fontweight='bold', 
             ha='center', va='center', rotation=90, color='darkblue')
    
    plt.suptitle('Confronto: Ricostruzioni VAE vs Immagini Originali per ogni Cifra', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.85)
    plt.show()


def calculate_elbo(model, data_loader, device, num_samples=1000):
    """
    Calcola l'Evidence Lower BOund (ELBO) del modello VAE.
    
    Args:
        model: Modello VAE addestrato
        data_loader: DataLoader da cui prendere i campioni
        device: Device su cui eseguire l'inferenza
        num_samples (int): Numero di campioni da utilizzare
        
    Returns:
        dict: Dizionario con ELBO totale, reconstruction loss e KL divergence
    """
    model.eval()
    total_elbo = 0
    total_recon_loss = 0
    total_kl_div = 0
    n_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            if n_samples >= num_samples:
                break
                
            data = data.to(device)
            batch_size = data.size(0)
            
            # Forward pass
            recon_data, mu, logvar = model(data)
            
            # Reconstruction loss (negative log likelihood)
            recon_loss = F.binary_cross_entropy(recon_data, data.view(-1, 784), reduction='sum')
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # ELBO = -reconstruction_loss - KL_divergence
            elbo = -recon_loss - kl_div
            
            total_elbo += elbo.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            n_samples += batch_size
    
    # Normalizza per il numero di campioni
    avg_elbo = total_elbo / n_samples
    avg_recon_loss = total_recon_loss / n_samples
    avg_kl_div = total_kl_div / n_samples
    
    return {
        'elbo': avg_elbo,
        'reconstruction_loss': avg_recon_loss,
        'kl_divergence': avg_kl_div
    }


def analyze_kl_divergence(model, data_loader, device, num_samples=1000):
    """
    Analizza la KL divergence per ogni dimensione latente.
    
    Args:
        model: Modello VAE addestrato
        data_loader: DataLoader da cui prendere i campioni
        device: Device su cui eseguire l'inferenza
        num_samples (int): Numero di campioni da utilizzare
        
    Returns:
        dict: Statistiche sulla KL divergence
    """
    model.eval()
    all_mu = []
    all_logvar = []
    n_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            if n_samples >= num_samples:
                break
                
            data = data.to(device)
            batch_size = data.size(0)
            
            # Ottieni mu e logvar
            mu, logvar = model.encode(data.view(-1, 784))
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            n_samples += batch_size
    
    # Concatena tutti i batch
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    
    # Calcola KL divergence per ogni dimensione
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    
    # Statistiche
    kl_mean = kl_per_dim.mean(dim=0)
    kl_std = kl_per_dim.std(dim=0)
    total_kl = kl_per_dim.sum(dim=1).mean()
    
    return {
        'kl_per_dimension': kl_per_dim.mean(dim=0).numpy(),
        'kl_std_per_dimension': kl_std.numpy(),
        'total_kl_mean': total_kl.item(),
        'mu_mean': all_mu.mean(dim=0).numpy(),
        'mu_std': all_mu.std(dim=0).numpy(),
        'logvar_mean': all_logvar.mean(dim=0).numpy(),
        'logvar_std': all_logvar.std(dim=0).numpy()
    }


def detect_posterior_collapse(model, data_loader, device, threshold=0.01, num_samples=1000):
    """
    Rileva il posterior collapse analizzando la KL divergence per dimensione.
    
    Args:
        model: Modello VAE addestrato
        data_loader: DataLoader da cui prendere i campioni
        device: Device su cui eseguire l'inferenza
        threshold (float): Soglia sotto la quale si considera collapsed una dimensione
        num_samples (int): Numero di campioni da utilizzare
        
    Returns:
        dict: Informazioni sul posterior collapse
    """
    kl_analysis = analyze_kl_divergence(model, data_loader, device, num_samples)
    kl_per_dim = kl_analysis['kl_per_dimension']
    
    # Identifica dimensioni collapsed
    collapsed_dims = kl_per_dim < threshold
    n_collapsed = collapsed_dims.sum()
    n_active = len(kl_per_dim) - n_collapsed
    
    # Calcola utilizzo medio delle dimensioni attive
    active_dims_kl = kl_per_dim[~collapsed_dims]
    avg_active_kl = active_dims_kl.mean() if len(active_dims_kl) > 0 else 0
    
    return {
        'n_collapsed_dimensions': int(n_collapsed),
        'n_active_dimensions': int(n_active),
        'total_dimensions': len(kl_per_dim),
        'collapse_ratio': float(n_collapsed / len(kl_per_dim)),
        'collapsed_dimensions': collapsed_dims,
        'kl_per_dimension': kl_per_dim,
        'avg_active_kl': float(avg_active_kl),
        'threshold': threshold
    }
