import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.vae import VAE
from utils.dataloader import get_mnist_dataset
from utils.device import device


def loss_function(x, y, mu, std):
    recon_loss = F.binary_cross_entropy(x, y.view(-1, 196), reduction="sum")
    kl_div = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
    return recon_loss + kl_div, recon_loss, kl_div

train_dataloader, test_dataloader = get_mnist_dataset()
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", factor=0.5, patience=5, threshold=0.001, cooldown=0, min_lr=0.0001
)

epochs = 20
train_losses, kl_losses, recon_losses = [], [], []

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, std = model(data)
        loss, recon, kl = loss_function(recon_batch, data, mu, std)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_losses.append(recon.item() / data.size(0))
        kl_losses.append(kl.item() / data.size(0))

    avg_loss = total_loss / len(train_dataloader.dataset)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

# Evaluation loop
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_dataloader:
        data = data.to(device)
        recon, mu, std = model(data)
        loss, _, _ = loss_function(recon, data, mu, std)
        test_loss += loss.item()

test_loss /= len(test_dataloader.dataset)
print(f"Test loss: {test_loss:.4f}")


model.eval()
with torch.no_grad():
    for i in range(4):
        data, _ = next(iter(test_dataloader))
        data = data.to(device)
        recon, _, _ = model(data)
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        for j in range(5):
            ax[0, j].imshow(data[j][0].cpu(), cmap="gray")
            ax[0, j].set_title("Original")
            ax[1, j].imshow(recon[j].view(14, 14).cpu(), cmap="gray")
            ax[1, j].set_title("Reconstructed")
        plt.show()
