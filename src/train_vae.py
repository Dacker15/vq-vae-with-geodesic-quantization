import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os


def vae_loss(recon_x, x, mu, logvar):
    """
    Calcola la loss function per il VAE.
    
    Args:
        recon_x: Immagini ricostruite
        x: Immagini originali
        mu: Media della distribuzione latente
        logvar: Log-varianza della distribuzione latente
        
    Returns:
        torch.Tensor: Loss totale (BCE + KLD)
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae(model, train_loader, device, epochs=64, learning_rate=1e-3, latent_dim=None, save_loss_plot=True):
    """
    Addestra il modello VAE.
    
    Args:
        model: Modello VAE da addestrare
        train_loader: DataLoader per il training
        device: Device su cui eseguire il training
        epochs (int): Numero di epoche
        learning_rate (float): Learning rate per l'optimizer
        latent_dim (int): Dimensione latente (per il nome del file del plot)
        save_loss_plot (bool): Se True, salva il plot della loss
        
    Returns:
        list: Lista delle loss per ogni epoca
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = train_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f'Epoca {epoch}: Loss medio = {avg_loss:.4f}')
    
    # Salva il plot della loss se richiesto
    if save_loss_plot and latent_dim is not None:
        save_training_loss_plot(epoch_losses, latent_dim)
    
    return epoch_losses


def save_training_loss_plot(epoch_losses, latent_dim):
    """
    Salva il plot dell'andamento della loss durante il training.
    
    Args:
        epoch_losses (list): Lista delle loss per ogni epoca
        latent_dim (int): Dimensione latente per il nome del file
    """
    # Crea la cartella se non esiste
    plots_dir = f'output/vae/training_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Disabilita la visualizzazione interattiva
    plt.ioff()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title(f'Andamento della Loss durante il Training - VAE {latent_dim}D', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Loss Media', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi annotazioni
    min_loss_epoch = epoch_losses.index(min(epoch_losses)) + 1
    min_loss_value = min(epoch_losses)
    plt.annotate(f'Min Loss: {min_loss_value:.4f}\nEpoca: {min_loss_epoch}', 
                xy=(min_loss_epoch, min_loss_value), 
                xytext=(min_loss_epoch + len(epoch_losses) * 0.1, min_loss_value + (max(epoch_losses) - min_loss_value) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Salva il plot
    plot_path = os.path.join(plots_dir, f'training_loss_{latent_dim}d.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot della loss salvato in: {plot_path}")
    
    # Riabilita la visualizzazione interattiva
    plt.ion()