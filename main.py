import torch
from src.models.vae import VAE
from src.utils.dataloader import get_mnist_dataloaders
from src.train_vae import train_vae
from src.utils.analysis import (
    get_latent_representations, 
    visualize_pca, 
    visualize_umap, 
    compare_pca_umap,
    analyze_umap_parameters,
    plot_reconstructions,
    interpolate_between_digits,
    sample_from_latent_space
)
from src.utils.device import device


def main():
    """Script principale per addestrare il VAE e analizzare lo spazio latente."""
    
    # Configurazione device
    print(f"Usando device: {device}")
    
    # Caricamento dati
    print("Caricamento dei dati MNIST...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # Inizializzazione modello
    print("Inizializzazione del modello VAE...")
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    is_vae_trained = False

    try:
        model.load_state_dict(torch.load('output/vae_model.pth', map_location=device))
        print("Modello caricato con successo.")
        is_vae_trained = True
    except FileNotFoundError:
        print("Nessun modello trovato, inizializzazione di un nuovo modello.")

    if not is_vae_trained:
        # Training
        print("Training del VAE...")
        epoch_losses = train_vae(
            model=model, 
            train_loader=train_loader, 
            device=device, 
            epochs=10, 
            learning_rate=1e-3
        )

        print('Salvataggio del modello in output...')
        torch.save(model.state_dict(), 'output/vae_model.pth')
    else:
        print("Modello già addestrato, saltando il training.")
    
    # Estrazione rappresentazioni latenti
    print("\nEstraendo rappresentazioni latenti...")
    latents, labels = get_latent_representations(
        model=model, 
        data_loader=test_loader, 
        device=device, 
        num_samples=2000
    )
    
    print(f"Shape delle rappresentazioni latenti: {latents.shape}")
    print(f"Shape delle labels: {labels.shape}")
    
    # Visualizzazioni
    print("\nVisualizzando con PCA...")
    pca_model = visualize_pca(latents, labels)
    
    print("\nVisualizzando con UMAP...")
    umap_model = visualize_umap(latents, labels)
    
    print("\nConfronto PCA vs UMAP...")
    compare_pca_umap(latents, labels)
    
    # Analisi parametri UMAP
    print("\nAnalisi parametri UMAP...")
    analyze_umap_parameters(latents, labels)
    
    # Visualizzazioni aggiuntive
    print("\nVisualizzando esempi di ricostruzione...")
    plot_reconstructions(model, test_loader, device, num_samples=8)
    
    print("\nCreando interpolazione tra cifre...")
    interpolate_between_digits(model, test_loader, device, digit1=0, digit2=8, num_steps=10)
    interpolate_between_digits(model, test_loader, device, digit1=3, digit2=7, num_steps=10)
    
    print("\nGenerando campioni casuali...")
    sample_from_latent_space(model, device, num_samples=16)
    
    print("\nAnalisi completata!")
    
    return model, epoch_losses, latents, labels


if __name__ == "__main__":
    model, losses, latents, labels = main()