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

def get_finetune_model(checkpoint_path):
    model = get_model()

    state_dict = torch.load(
        checkpoint_path, 
        map_location="cpu"
    )
    model.load_state_dict(state_dict)

    for param in model.layer4.parameters():
        param.requires_grad = True

    return model



if __name__ == "__main__":
    model = get_model()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    