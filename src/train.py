import torch 
import torch.nn as nn
import torch.optim as optim
from src.data import get_dataloaders
from src.model import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def train():
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, _ = get_dataloaders()
    model = get_model()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.fc.parameters(), 
        lr=0.001
    )

    num_epochs = 10
    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # train loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # validation loop
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(), 
                "outputs/checkpoints/best_model.pth"
            )

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {epoch_loss:.4f} - "
            f"Train Acc: {train_accuracy:.2f}% - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    return train_losses, train_accuracies, val_losses, val_accuracies

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig("outputs/figures/loss_curve.png")
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.savefig("outputs/figures/accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    train_losses, train_accuracies, val_losses, val_accuracies = train()

    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)