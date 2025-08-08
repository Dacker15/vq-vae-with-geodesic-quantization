import torch
from src.models.vae import VAE
from src.utils.dataloader import get_mnist_dataloaders
from src.train_vae import train_vae
from src.utils.analysis import (
    get_latent_representations, 
    visualize_pca, 
    visualize_umap, 
    compare_pca_umap,
    analyze_umap_parameters
)


def main():
    """Script principale per addestrare il VAE e analizzare lo spazio latente."""
    
    # Configurazione device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Caricamento dati
    print("Caricamento dei dati MNIST...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # Inizializzazione modello
    print("Inizializzazione del modello VAE...")
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    
    # Training
    print("Training del VAE...")
    epoch_losses = train_vae(
        model=model, 
        train_loader=train_loader, 
        device=device, 
        epochs=10, 
        learning_rate=1e-3
    )
    
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
    
    print("\nAnalisi completata!")
    
    return model, epoch_losses, latents, labels


if __name__ == "__main__":
    model, losses, latents, labels = main()