import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset


def resize_and_binarize(images):
    resized = np.array([cv2.resize(img.astype("float32"), (14, 14)) for img in images])
    binarized = np.where(resized > 128, 1, 0).astype(np.float32)
    return binarized

def get_mnist_dataset(train_batch_size=32, test_batch_size=100):
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    xtrain = trainset.data.numpy()
    ytrain = trainset.targets.numpy()
    x_test_pre = testset.data[:1000].numpy()
    y_test = testset.targets[:1000].numpy()

    count = np.zeros(10)
    idx = []
    for i in range(len(ytrain)):
        label = ytrain[i]
        if count[label] < 1000:
            count[label] += 1
            idx.append(i)

    x_train_pre = xtrain[idx]
    y_train = ytrain[idx]

    x_train = resize_and_binarize(x_train_pre)
    x_val = resize_and_binarize(x_test_pre)

    train_tensor = TensorDataset(torch.tensor(x_train).unsqueeze(1), torch.tensor(y_train))
    test_tensor = TensorDataset(torch.tensor(x_val).unsqueeze(1), torch.tensor(y_test))

    trainloader = DataLoader(train_tensor, batch_size=train_batch_size, shuffle=True)
    testloader = DataLoader(test_tensor, batch_size=test_batch_size, shuffle=False)

    return trainloader, testloader
