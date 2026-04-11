from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training = datasets.ImageFolder('data/training/', transform=transform)
    test = datasets.ImageFolder('data/testing/', transform=transform)

    train_loader = DataLoader(training, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)

    return train_loader, test_loader
