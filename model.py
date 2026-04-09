import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        try:
            model = models.resnet18(weights=None)
        except Exception:
            model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
