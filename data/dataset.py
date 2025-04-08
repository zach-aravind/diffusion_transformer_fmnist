import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_fashion_mnist_dataset(img_size):
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
    ])
    
    # Load dataset
    dataset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
    
    return dataset

def get_dataloader(dataset, batch_size, num_workers=4):
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return dataloader