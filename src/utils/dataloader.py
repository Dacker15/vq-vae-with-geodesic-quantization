import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def get_mnist_dataloaders(batch_size=128):
    """
    Carica i dataset MNIST e crea i DataLoader per training e test.
    
    Args:
        batch_size (int): Dimensione del batch
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Trasformazioni
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader