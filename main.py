import torch
import os
from src.models.vae import VAE
from src.utils.dataloader import get_mnist_dataloaders
from src.train_vae import train_vae
from src.utils.analysis import (
    get_latent_representations, 
    visualize_pca, 
    visualize_umap, 
    compare_pca_umap,
    analyze_umap_parameters,
    sample_from_latent_space,
    calculate_elbo,
    analyze_kl_divergence,
    detect_posterior_collapse,
    visualize_kl_analysis,
    compare_models_metrics
)
from src.utils.device import device


def train_and_evaluate_model(latent_dim, train_loader, test_loader, epochs=10):
    """
    Addestra e valuta un modello VAE con una specifica dimensione latente.
    
    Args:
        latent_dim (int): Dimensione dello spazio latente
        train_loader: DataLoader per il training
        test_loader: DataLoader per il test
        epochs (int): Numero di epoche di training
        
    Returns:
        dict: Dizionario con il modello e le metriche calcolate
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODELLO CON {latent_dim} DIMENSIONI LATENTI")
    print(f"{'='*60}")
    
    # Inizializzazione modello
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
    model_path = f'output/vae_model_{latent_dim}d.pth'
    
    # Controllo se il modello esiste già
    if os.path.exists(model_path):
        print(f"Caricamento modello esistente: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training nuovo modello con {latent_dim} dimensioni latenti...")
        epoch_losses = train_vae(
            model=model, 
            train_loader=train_loader, 
            device=device, 
            epochs=epochs, 
            learning_rate=1e-3
        )

        print(f"Loss per epoca: {epoch_losses}")

        # Salva il modello
        print(f"Salvataggio modello: {model_path}")
        torch.save(model.state_dict(), model_path)
    
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
    
    return {
        'model': model,
        'latent_dim': latent_dim,
        'elbo': elbo_results,
        'kl_analysis': kl_results,
        'collapse': collapse_results
    }


def main():
    """Script principale per testare VAE con diverse dimensioni latenti."""
    
    # Configurazione
    print(f"Usando device: {device}")
    
    # Array delle dimensioni latenti da testare
    latent_dimensions = [2, 5, 10, 20, 32, 64]
    print(f"Dimensioni latenti da testare: {latent_dimensions}")
    
    # Creazione cartella output se non esiste
    os.makedirs('output', exist_ok=True)
    
    # Caricamento dati
    print("Caricamento dei dati MNIST...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # Dizionario per memorizzare tutti i risultati
    all_results = {}
    
    # Training e valutazione per ogni dimensione latente
    for latent_dim in latent_dimensions:
        result = train_and_evaluate_model(
            latent_dim=latent_dim,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=10
        )
        all_results[latent_dim] = result
    
    # Confronto e visualizzazione risultati
    print(f"\n{'='*80}")
    print("CONFRONTO RISULTATI PER TUTTE LE DIMENSIONI")
    print(f"{'='*80}")
    
    # Confronto metriche
    compare_models_metrics(all_results)
    
    for dim in latent_dimensions:
        print(f"\n{'='*60}")
        print(f"ANALISI DETTAGLIATA - MODELLO {dim}D")
        print(f"{'='*60}")
        
        model = all_results[dim]['model']
        
        # Estrazione rappresentazioni latenti
        print("Estraendo rappresentazioni latenti...")
        latents, labels = get_latent_representations(
            model=model, 
            data_loader=test_loader, 
            device=device, 
            num_samples=2000
        )
        
        print(f"Shape delle rappresentazioni latenti: {latents.shape}")
        
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

    # Visualizzazione comparativa dell'analisi KL
    print(f"\n{'='*60}")
    print("ANALISI KL DIVERGENCE COMPARATIVA")
    print(f"{'='*60}")
    
    kl_results_list = [all_results[dim]['kl_analysis'] for dim in latent_dimensions]
    visualize_kl_analysis(kl_results_list, latent_dimensions)
    
    print("\n" + "="*80)
    print("ANALISI COMPLETATA!")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()