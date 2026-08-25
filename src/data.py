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
        transform=None
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=generator
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
        transform=val_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=val_transform
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

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    test_images, test_labels = next(iter(test_loader))

    print(test_images.shape)
    print(test_labels.shape)