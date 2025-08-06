import torch
import torch.nn as nn

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
