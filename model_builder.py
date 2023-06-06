import torch # noqa 5501
import torchvision # noqa 5501
from torch import nn # noqa 5501
from torchinfo import summary # noqa 5501
from torch.utils.tensorboard import SummaryWriter # noqa 5501

import utils


def create_effnetb2(num_of_classes: int, device="cuda"):

    model_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = model_weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=model_weights).to(device) # noqa 5501

    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds(42)

    # # Set cllasifier to suit problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408,
                  out_features=num_of_classes,
                  bias=True).to(device))

    model.name = "effnetb2"
    return model, transforms


def create_resnet50(num_of_classes: int, device="cuda"):

    model_weights = torchvision.models.ResNet50_Weights.DEFAULT
    transforms = model_weights.transforms()
    model = torchvision.models.resnet50(weights=model_weights).to(device)

    # # Freeze layers
    layers = [model.layer1, model.layer2,
              model.layer3, model.layer4]
    for layer in layers:
        layer.requires_grad_(False)
    model.conv1.requires_grad_(False)

    utils.set_seeds(42)

    # # Set cllasifier to suit problem
    model.fc = nn.Linear(in_features=2048,
                         out_features=num_of_classes,
                         bias=True).to(device)

    model.name = "resnet50"
    return model, transforms
