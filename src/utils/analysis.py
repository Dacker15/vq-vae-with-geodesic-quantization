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


def visualize_kl_analysis(kl_analysis_results, latent_dims):
    """
    Visualizza i risultati dell'analisi KL per diverse dimensioni latenti.
    
    Args:
        kl_analysis_results: Lista di risultati dell'analisi KL per ogni dimensione
        latent_dims: Lista delle dimensioni latenti testate
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # KL divergence per dimensione latente
    axes[0, 0].bar(range(len(latent_dims)), [r['total_kl_mean'] for r in kl_analysis_results])
    axes[0, 0].set_xlabel('Dimensioni Latenti')
    axes[0, 0].set_ylabel('KL Divergence Media')
    axes[0, 0].set_title('KL Divergence Totale per Dimensione Latente')
    axes[0, 0].set_xticks(range(len(latent_dims)))
    axes[0, 0].set_xticklabels(latent_dims)
    
    # Varianza delle medie
    axes[0, 1].bar(range(len(latent_dims)), [r['mu_std'].mean() for r in kl_analysis_results])
    axes[0, 1].set_xlabel('Dimensioni Latenti')
    axes[0, 1].set_ylabel('Std Media di μ')
    axes[0, 1].set_title('Variabilità delle Medie Latenti')
    axes[0, 1].set_xticks(range(len(latent_dims)))
    axes[0, 1].set_xticklabels(latent_dims)
    
    # Heatmap KL per ogni dimensione
    max_dim = max(latent_dims)
    kl_matrix = np.zeros((len(latent_dims), max_dim))
    
    for i, result in enumerate(kl_analysis_results):
        kl_per_dim = result['kl_per_dimension']
        kl_matrix[i, :len(kl_per_dim)] = kl_per_dim
    
    im = axes[1, 0].imshow(kl_matrix, aspect='auto', cmap='viridis')
    axes[1, 0].set_xlabel('Dimensione Latente')
    axes[1, 0].set_ylabel('Configurazione del Modello')
    axes[1, 0].set_title('KL Divergence per Dimensione')
    axes[1, 0].set_yticks(range(len(latent_dims)))
    axes[1, 0].set_yticklabels([f'{dim}D' for dim in latent_dims])
    plt.colorbar(im, ax=axes[1, 0])
    
    # Distribuzione delle log-varianze
    for i, result in enumerate(kl_analysis_results):
        logvar_mean = result['logvar_mean']
        axes[1, 1].plot(logvar_mean, label=f'{latent_dims[i]}D', alpha=0.7)
    
    axes[1, 1].set_xlabel('Dimensione Latente')
    axes[1, 1].set_ylabel('Log-Varianza Media')
    axes[1, 1].set_title('Distribuzione Log-Varianza')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def compare_models_metrics(results_dict):
    """
    Confronta le metriche tra diversi modelli con dimensioni latenti diverse.
    
    Args:
        results_dict: Dizionario con i risultati per ogni dimensione latente
    """
    latent_dims = list(results_dict.keys())
    
    # Estrai metriche
    elbos = [results_dict[dim]['elbo']['elbo'] for dim in latent_dims]
    recon_losses = [results_dict[dim]['elbo']['reconstruction_loss'] for dim in latent_dims]
    kl_divs = [results_dict[dim]['elbo']['kl_divergence'] for dim in latent_dims]
    collapse_ratios = [results_dict[dim]['collapse']['collapse_ratio'] for dim in latent_dims]
    active_dims = [results_dict[dim]['collapse']['n_active_dimensions'] for dim in latent_dims]
    
    # Crea il plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ELBO
    axes[0, 0].plot(latent_dims, elbos, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Dimensioni Latenti')
    axes[0, 0].set_ylabel('ELBO')
    axes[0, 0].set_title('Evidence Lower BOund')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(latent_dims, recon_losses, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Dimensioni Latenti')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Perdita di Ricostruzione')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[0, 2].plot(latent_dims, kl_divs, 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Dimensioni Latenti')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].set_title('Divergenza di Kullback-Leibler')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Posterior Collapse Ratio
    axes[1, 0].plot(latent_dims, collapse_ratios, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Dimensioni Latenti')
    axes[1, 0].set_ylabel('Rapporto di Collasso')
    axes[1, 0].set_title('Posterior Collapse Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dimensioni Attive
    axes[1, 1].plot(latent_dims, active_dims, 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Dimensioni Latenti')
    axes[1, 1].set_ylabel('Dimensioni Attive')
    axes[1, 1].set_title('Numero di Dimensioni Attive')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Efficienza delle dimensioni (Active dims / Total dims)
    efficiency = [active_dims[i] / latent_dims[i] for i in range(len(latent_dims))]
    axes[1, 2].plot(latent_dims, efficiency, 'yo-', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Dimensioni Latenti')
    axes[1, 2].set_ylabel('Efficienza')
    axes[1, 2].set_title('Efficienza delle Dimensioni (Attive/Totali)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stampa sommario
    print("\n" + "="*80)
    print("SOMMARIO CONFRONTO MODELLI")
    print("="*80)
    
    for i, dim in enumerate(latent_dims):
        print(f"\nModello con {dim} dimensioni latenti:")
        print(f"  ELBO: {elbos[i]:.4f}")
        print(f"  Reconstruction Loss: {recon_losses[i]:.4f}")
        print(f"  KL Divergence: {kl_divs[i]:.4f}")
        print(f"  Dimensioni Attive: {active_dims[i]}/{dim} ({efficiency[i]:.2%})")
        print(f"  Posterior Collapse: {collapse_ratios[i]:.2%}")
    
    # Trova il miglior modello per ogni metrica
    best_elbo_idx = np.argmax(elbos)
    best_efficiency_idx = np.argmax(efficiency)
    min_collapse_idx = np.argmin(collapse_ratios)
    
    print(f"\n" + "-"*80)
    print("MIGLIORI MODELLI:")
    print(f"  Miglior ELBO: {latent_dims[best_elbo_idx]} dimensioni (ELBO: {elbos[best_elbo_idx]:.4f})")
    print(f"  Maggiore Efficienza: {latent_dims[best_efficiency_idx]} dimensioni ({efficiency[best_efficiency_idx]:.2%})")
    print(f"  Minor Collapse: {latent_dims[min_collapse_idx]} dimensioni ({collapse_ratios[min_collapse_idx]:.2%})")
    print("-"*80)