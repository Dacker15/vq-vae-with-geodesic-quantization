import torch
import torch.nn.functional as F
import torch.optim as optim


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


def train_vae(model, train_loader, device, epochs=10, learning_rate=1e-3):
    """
    Addestra il modello VAE.
    
    Args:
        model: Modello VAE da addestrare
        train_loader: DataLoader per il training
        device: Device su cui eseguire il training
        epochs (int): Numero di epoche
        learning_rate (float): Learning rate per l'optimizer
        
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
    
    return epoch_losses