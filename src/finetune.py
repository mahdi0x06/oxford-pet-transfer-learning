import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.data import get_dataloaders
from src.model import get_finetune_model
import matplotlib.pyplot as plt

def set_frozen_backbone_eval(model):
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()


def finetune():
    train_loader, val_loader, _ = get_dataloaders()

    model = get_finetune_model(
        "outputs/checkpoints/best_model.pth"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {
            "params": model.layer4.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-3
        }
    ])

    num_epochs = 5

    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        set_frozen_backbone_eval(model)

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Fine-tune {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "outputs/checkpoints/best_finetuned_model.pth"
            )


    return (
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies
    )

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-Tuning: Training and Validation Loss")
    plt.legend()

    plt.savefig("outputs/figures/finetune_loss_curve.png")
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Fine-Tuning: Training and Validation Accuracy")
    plt.legend()

    plt.savefig("outputs/figures/finetune_accuracy_curve.png")
    plt.close()

     
if __name__ == "__main__":
    train_losses, train_accuracies, val_losses, val_accuracies = finetune()

    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)