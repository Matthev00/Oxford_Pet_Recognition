import torchvision
from torch import nn
import torch


def create_effnetb2(num_of_classes: int, device="cuda"):

    model_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = model_weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=model_weights).to(device) # noqa 5501

    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds(42)

    # # Set cllasifier to suit problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408,
                  out_features=num_of_classes,
                  bias=True).to(device))

    model.name = "effnetb2"
    return model, transforms


def set_seeds(seed: int = 42):

    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
