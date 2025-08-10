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


def get_mnist_single_batch(max_samples=8000, split='test'):
    """
    Carica l'intero dataset MNIST in un singolo batch.
    
    Args:
        max_samples (int): Numero massimo di campioni da caricare
        split (str): 'train' o 'test'
        
    Returns:
        tuple: (data_tensor, labels_tensor) 
    """
    # Trasformazioni
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset
    if split == 'train':
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
    else:
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    
    # Limita il numero di campioni se specificato
    if max_samples and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Carica tutto in un singolo batch
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(dataloader))
    
    print(f"Caricati {len(data)} campioni MNIST ({split}) in un singolo batch")
    
    return data, labels