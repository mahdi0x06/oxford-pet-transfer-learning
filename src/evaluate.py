import torch

from src.data import get_dataloaders
from src.model import get_model
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate():
    _, _, test_loader = get_dataloaders()
    model = get_model()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    state_dict = torch.load(
        "outputs/checkpoints/best_model.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    class_names = test_loader.dataset.classes

    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(
        all_labels, 
        all_predictions, 
        normalize="true"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(18, 18))
    display.plot(
        ax=ax, 
        xticks_rotation=90, 
        values_format=".2f", 
        colorbar=False
    )

    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "outputs/figures/confusion_matrix.png",
        dpi=200
    )

    plt.close()

    print("\nClassification Report:")
    print(report)

    print(f"Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall:    {recall * 100:.2f}%")
    print(f"Macro F1:        {f1 * 100:.2f}%")

if __name__ == "__main__":
    evaluate()