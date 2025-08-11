import torch
import os
from src.models.vae import VAE
from src.utils.dataloader import get_mnist_dataloaders
from src.train_vae import train_vae
from src.utils.device import device


def train_model(latent_dim, train_loader, epochs=64):
    """
    Addestra un modello VAE con una specifica dimensione latente.
    
    Args:
        latent_dim (int): Dimensione dello spazio latente
        train_loader: DataLoader per il training
        epochs (int): Numero di epoche di training
        
    Returns:
        str: Percorso del modello salvato
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODELLO CON {latent_dim} DIMENSIONI LATENTI")
    print(f"{'='*60}")
    
    # Inizializzazione modello
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
    model_path = f'output/vae/vae_model_{latent_dim}d.pth'
    
    # Controllo se il modello esiste già
    if os.path.exists(model_path):
        print(f"Modello già esistente trovato: {model_path}")
        print("Salto il training...")
        return model_path
    
    print(f"Training nuovo modello con {latent_dim} dimensioni latenti...")
    print(f"Epoche: {epochs}")
    
    # Training del modello
    epoch_losses = train_vae(
        model=model, 
        train_loader=train_loader, 
        device=device, 
        epochs=epochs, 
        learning_rate=1e-3,
        latent_dim=latent_dim,
        save_loss_plot=True
    )

    print(f"Training completato! Loss finale: {epoch_losses[-1]:.4f}")

    # Salva il modello
    print(f"Salvataggio modello: {model_path}")
    torch.save(model.state_dict(), model_path)
    
    return model_path


def main():
    """Script principale per addestrare VAE con diverse dimensioni latenti."""
    
    # Configurazione
    print(f"Usando device: {device}")
    
    # Array delle dimensioni latenti da testare
    latent_dimensions = [20, 32, 64]
    print(f"Dimensioni latenti da testare: {latent_dimensions}")
    
    # Creazione cartelle output se non esistono
    os.makedirs('output/vae', exist_ok=True)
    os.makedirs('output/vae/training_plots', exist_ok=True)
    
    # Caricamento dati
    print("Caricamento dei dati MNIST...")
    train_loader, _ = get_mnist_dataloaders(batch_size=128)
    
    # Lista per memorizzare i percorsi dei modelli addestrati
    trained_models = []
    
    # Training per ogni dimensione latente
    for latent_dim in latent_dimensions:
        model_path = train_model(
            latent_dim=latent_dim,
            train_loader=train_loader,
            epochs=64
        )
        trained_models.append((latent_dim, model_path))
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETATO PER TUTTE LE DIMENSIONI")
    print(f"{'='*80}")
    
    print("\nModelli addestrati:")
    for latent_dim, model_path in trained_models:
        print(f"  - {latent_dim}D: {model_path}")
    
    print(f"\nPlot delle loss salvati in: output/vae/training_plots/")
    
    print("\n" + "="*80)
    print("TRAINING AUTOMATIZZATO COMPLETATO!")
    print("="*80)
    
    return trained_models


if __name__ == "__main__":
    trained_models = main()