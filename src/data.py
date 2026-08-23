import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_dataloaders():

    full_dataset = datasets.OxfordIIITPet(
        root="data", 
        split="trainval", 
        target_types="category", 
        download=True, 
        transforms=None
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size]
    )

    train_dataset = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=False,
        transform=train_transform
    )

    val_dataset = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=False,
        transforms=val_transform
    )

    train_dataset = Subset(
        train_dataset, 
        train_subset.indices
    )

    val_dataset = Subset(
        val_dataset, 
        val_subset.indices
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False
    )
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)
    print(labels[:10])