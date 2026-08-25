from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch

def get_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37)

    return model






if __name__ == "__main__":
    model = get_model()
    