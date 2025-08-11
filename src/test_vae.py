import torch
import os
import matplotlib.pyplot as plt
from models.vae import VAE
from utils.dataloader import get_mnist_dataloaders
from utils.analysis import (
    get_latent_representations, 
    visualize_pca, 
    visualize_umap, 
    compare_pca_umap,
    analyze_umap_parameters,
    sample_from_latent_space,
    calculate_elbo,
    analyze_kl_divergence,
    detect_posterior_collapse,
)
from utils.device import device


def evaluate_model(latent_dim, test_loader, save_plots=True):
    """
    Valuta un modello VAE addestrato con una specifica dimensione latente.
    
    Args:
        latent_dim (int): Dimensione dello spazio latente
        test_loader: DataLoader per il test
        save_plots (bool): Se True, salva i grafici invece di mostrarli
        
    Returns:
        dict: Dizionario con le metriche calcolate
    """
    print(f"\n{'='*60}")
    print(f"VALUTAZIONE MODELLO CON {latent_dim} DIMENSIONI LATENTI")
    print(f"{'='*60}")
    
    # Caricamento modello
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
    model_path = f'output/vae/vae_model_{latent_dim}d.pth'
    
    if not os.path.exists(model_path):
        print(f"ERRORE: Modello non trovato: {model_path}")
        return None
        
    print(f"Caricamento modello: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Calcolo metriche
    print(f"Calcolo metriche per modello {latent_dim}D...")
    
    # ELBO
    elbo_results = calculate_elbo(model, test_loader, device, num_samples=2000)
    print(f"ELBO: {elbo_results['elbo']:.4f}")
    
    # KL Divergence Analysis
    kl_results = analyze_kl_divergence(model, test_loader, device, num_samples=2000)
    print(f"KL Divergence Media: {kl_results['total_kl_mean']:.4f}")
    
    # Posterior Collapse Detection
    collapse_results = detect_posterior_collapse(model, test_loader, device, 
                                               threshold=0.01, num_samples=2000)
    print(f"Dimensioni Collassate: {collapse_results['n_collapsed_dimensions']}/{latent_dim}")
    print(f"Rapporto di Collasso: {collapse_results['collapse_ratio']:.2%}")
    
    # Estrazione rappresentazioni latenti
    print("Estraendo rappresentazioni latenti...")
    latents, labels = get_latent_representations(
        model=model, 
        data_loader=test_loader, 
        device=device, 
        num_samples=2000
    )
    
    print(f"Shape delle rappresentazioni latenti: {latents.shape}")
    
    # Creazione cartella per i grafici
    plots_dir = f'output/vae/plots_{latent_dim}d'
    os.makedirs(plots_dir, exist_ok=True)
    
    if save_plots:
        # Disabilita la visualizzazione interattiva
        plt.ioff()
        
        print("Visualizzazione con PCA...")
        pca = visualize_pca_saved(latents, labels, plots_dir)
        
        print("Visualizzazione con UMAP...")
        umap_model = visualize_umap_saved(latents, labels, plots_dir)
        
        print("Confronto PCA vs UMAP...")
        compare_pca_umap_saved(latents, labels, plots_dir)
        
        print("\nAnalisi parametri UMAP...")
        analyze_umap_parameters_saved(latents, labels, plots_dir)

        print("Generando campioni casuali...")
        sample_from_latent_space_saved(model, test_loader, device, plots_dir)
        
        # Riabilita la visualizzazione interattiva
        plt.ion()
    else:
        print("Visualizzazione con PCA...")
        visualize_pca(latents, labels)
        
        print("Visualizzazione con UMAP...")
        visualize_umap(latents, labels)
        
        print("Confronto PCA vs UMAP...")
        compare_pca_umap(latents, labels)
        
        print("\nAnalisi parametri UMAP...")
        analyze_umap_parameters(latents, labels)

        print("Generando campioni casuali...")
        sample_from_latent_space(model, data_loader=test_loader, device=device)
    
    return {
        'model': model,
        'latent_dim': latent_dim,
        'elbo': elbo_results,
        'kl_analysis': kl_results,
        'collapse': collapse_results
    }


def visualize_pca_saved(latents, labels, plots_dir, n_components=2):
    """Versione di visualize_pca che salva i grafici invece di mostrarli."""
    from sklearn.decomposition import PCA
    import numpy as np
    
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
    plt.savefig(os.path.join(plots_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca


def visualize_umap_saved(latents, labels, plots_dir, n_neighbors=15, min_dist=0.1):
    """Versione di visualize_umap che salva i grafici invece di mostrarli."""
    import umap
    import numpy as np
    
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
    plt.savefig(os.path.join(plots_dir, 'umap_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return umap_model


def compare_pca_umap_saved(latents, labels, plots_dir):
    """Versione di compare_pca_umap che salva i grafici invece di mostrarli."""
    from sklearn.decomposition import PCA
    import umap
    
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
    plt.savefig(os.path.join(plots_dir, 'pca_umap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_umap_parameters_saved(latents, labels, plots_dir):
    """Versione di analyze_umap_parameters che salva i grafici invece di mostrarli."""
    import umap
    
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
    plt.savefig(os.path.join(plots_dir, 'umap_parameters_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def sample_from_latent_space_saved(model, data_loader, device, plots_dir):
    """Versione di sample_from_latent_space che salva i grafici invece di mostrarli."""
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
    plt.savefig(os.path.join(plots_dir, 'reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_kl_analysis_saved(kl_results_list, latent_dimensions, output_dir):
    """Versione di visualize_kl_analysis che salva i grafici invece di mostrarli."""
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # KL divergence per dimensione latente
    axes[0, 0].bar(range(len(latent_dimensions)), [r['total_kl_mean'] for r in kl_results_list])
    axes[0, 0].set_xlabel('Dimensioni Latenti')
    axes[0, 0].set_ylabel('KL Divergence Media')
    axes[0, 0].set_title('KL Divergence Totale per Dimensione Latente')
    axes[0, 0].set_xticks(range(len(latent_dimensions)))
    axes[0, 0].set_xticklabels(latent_dimensions)
    
    # Varianza delle medie
    axes[0, 1].bar(range(len(latent_dimensions)), [r['mu_std'].mean() for r in kl_results_list])
    axes[0, 1].set_xlabel('Dimensioni Latenti')
    axes[0, 1].set_ylabel('Std Media di μ')
    axes[0, 1].set_title('Variabilità delle Medie Latenti')
    axes[0, 1].set_xticks(range(len(latent_dimensions)))
    axes[0, 1].set_xticklabels(latent_dimensions)
    
    # Heatmap KL per ogni dimensione
    max_dim = max(latent_dimensions)
    kl_matrix = np.zeros((len(latent_dimensions), max_dim))
    
    for i, result in enumerate(kl_results_list):
        kl_per_dim = result['kl_per_dimension']
        kl_matrix[i, :len(kl_per_dim)] = kl_per_dim
    
    im = axes[1, 0].imshow(kl_matrix, aspect='auto', cmap='viridis')
    axes[1, 0].set_xlabel('Dimensione Latente')
    axes[1, 0].set_ylabel('Configurazione del Modello')
    axes[1, 0].set_title('KL Divergence per Dimensione')
    axes[1, 0].set_yticks(range(len(latent_dimensions)))
    axes[1, 0].set_yticklabels([f'{dim}D' for dim in latent_dimensions])
    plt.colorbar(im, ax=axes[1, 0])
    
    # Distribuzione delle log-varianze
    for i, result in enumerate(kl_results_list):
        logvar_mean = result['logvar_mean']
        axes[1, 1].plot(logvar_mean, label=f'{latent_dimensions[i]}D', alpha=0.7)
    
    axes[1, 1].set_xlabel('Dimensione Latente')
    axes[1, 1].set_ylabel('Log-Varianza Media')
    axes[1, 1].set_title('Distribuzione Log-Varianza')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_analysis_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compare_models_metrics_saved(results_dict, output_dir):
    """Versione di compare_models_metrics che salva i grafici invece di mostrarli."""
    import numpy as np
    
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
    plt.savefig(os.path.join(output_dir, 'models_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
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


def main():
    """Script principale per valutare VAE con diverse dimensioni latenti."""
    
    # Configurazione
    print(f"Usando device: {device}")
    
    # Array delle dimensioni latenti da testare
    latent_dimensions = [20, 32, 64]
    print(f"Dimensioni latenti da valutare: {latent_dimensions}")
    
    # Creazione cartelle output se non esistono
    os.makedirs('output/vae', exist_ok=True)
    os.makedirs('output/vae/evaluation', exist_ok=True)
    
    # Caricamento dati
    print("Caricamento dei dati MNIST...")
    _, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # Dizionario per memorizzare tutti i risultati
    all_results = {}
    
    # Valutazione per ogni dimensione latente
    for latent_dim in latent_dimensions:
        result = evaluate_model(
            latent_dim=latent_dim,
            test_loader=test_loader,
            save_plots=True
        )
        if result is not None:
            all_results[latent_dim] = result
    
    if len(all_results) == 0:
        print("ERRORE: Nessun modello trovato per la valutazione!")
        return
    
    # Confronto e visualizzazione risultati
    print(f"\n{'='*80}")
    print("CONFRONTO RISULTATI PER TUTTE LE DIMENSIONI")
    print(f"{'='*80}")
    
    # Disabilita la visualizzazione interattiva per i confronti
    plt.ioff()
    
    # Confronto metriche
    compare_models_metrics_saved(all_results, 'output/vae/evaluation')
    
    # Visualizzazione comparativa dell'analisi KL
    print(f"\n{'='*60}")
    print("ANALISI KL DIVERGENCE COMPARATIVA")
    print(f"{'='*60}")
    
    kl_results_list = [all_results[dim]['kl_analysis'] for dim in latent_dimensions if dim in all_results]
    available_dims = [dim for dim in latent_dimensions if dim in all_results]
    
    if len(kl_results_list) > 0:
        visualize_kl_analysis_saved(kl_results_list, available_dims, 'output/vae/evaluation')
    
    # Riabilita la visualizzazione interattiva
    plt.ion()
    
    print("\n" + "="*80)
    print("VALUTAZIONE COMPLETATA!")
    print(f"Grafici salvati in: output/vae/evaluation/")
    print(f"Grafici individuali salvati in: output/vae/plots_<dim>d/")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()
