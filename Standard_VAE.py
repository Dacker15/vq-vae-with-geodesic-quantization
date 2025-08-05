import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Download MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Numpy conversion
xtrain = trainset.data.numpy()
ytrain = trainset.targets.numpy()
x_val_pre = testset.data[:1000].numpy()
y_val = testset.targets[:1000].numpy()

# Sottocampionamento bilanciato (1000 per cifra)
count = np.zeros(10)
idx = []
for i in range(len(ytrain)):
    label = ytrain[i]
    if count[label] < 1000:
        count[label] += 1
        idx.append(i)

x_train_pre = xtrain[idx]
y_train = ytrain[idx]

# Resize a 14x14
def resize_and_binarize(images):
    resized = np.array([cv2.resize(img.astype('float32'), (14, 14)) for img in images])
    binarized = np.where(resized > 128, 1, 0).astype(np.float32)
    return binarized

x_train = resize_and_binarize(x_train_pre)
x_val = resize_and_binarize(x_val_pre)

# Costruzione dataloader
train_tensor = TensorDataset(torch.tensor(x_train).unsqueeze(1), torch.tensor(y_train))
val_tensor = TensorDataset(torch.tensor(x_val).unsqueeze(1), torch.tensor(y_val))

trainloader = DataLoader(train_tensor, batch_size=32, shuffle=True)
testloader = DataLoader(val_tensor, batch_size=100, shuffle=False)

# Definizione VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(196, 128)
        self.fc21 = nn.Linear(128, 8)
        self.fc22 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 128)
        self.fc4 = nn.Linear(128, 196)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def sampling(self, mu, std):
        eps1 = torch.randn_like(std)
        eps2 = torch.randn_like(std)
        return 0.5 * ((eps1 * std + mu) + (eps2 * std + mu))

    def decoder(self, z):
        h = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, std = self.encoder(x.view(-1, 196))
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, 
                             patience=5, threshold=0.001, cooldown=0,
                             min_lr=0.0001, verbose=True)


# Funzione di loss per il VAE (ELBO)
def loss_function(x, y, mu, std):
    recon_loss = F.binary_cross_entropy(x, y.view(-1, 196), reduction='sum')
    kl_div = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
    return recon_loss + kl_div, recon_loss, kl_div

# Training loop
epochs = 20
train_losses, kl_losses, recon_losses = [], [], []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, std = model(data)
        loss, recon, kl = loss_function(recon_batch, data, mu, std)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        recon_losses.append(recon.item() / data.size(0))
        kl_losses.append(kl.item() / data.size(0))

    avg_loss = total_loss / len(trainloader.dataset)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in testloader:
        data = data.to(device)
        recon, mu, std = model(data)
        loss, _, _ = loss_function(recon, data, mu, std)
        test_loss += loss.item()

test_loss /= len(testloader.dataset)
print(f"Test loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for i in range(4):  # mostra 4 batch
        data, _ = next(iter(testloader))
        data = data.to(device)
        recon, _, _ = model(data)
        for j in range(5):  # 5 immagini per batch
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(data[j][0].cpu(), cmap='gray')
            ax[0].set_title("Original")
            ax[1].imshow(recon[j].view(14, 14).cpu(), cmap='gray')
            ax[1].set_title("Reconstructed")
            plt.show()
