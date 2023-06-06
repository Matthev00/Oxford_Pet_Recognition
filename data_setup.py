from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from utils import set_seeds
import torch
import ssl


def download_data(data_path: Path,
                  train_transform: transforms = None,
                  test_transform: transforms = None):

    train_data = datasets.OxfordIIITPet(root=data_path,
                                        split="trainval",
                                        target_types="category",
                                        transform=train_transform,
                                        download=True)

    test_data = datasets.OxfordIIITPet(root=data_path,
                                       split="test",
                                       target_types="category",
                                       transform=test_transform,
                                       download=True)

    return train_data, test_data


def split_data(dataset: datasets,
               split: float = 0.2):

    length1 = int(split * len(dataset))
    length2 = len(dataset) - length1

    set_seeds(42)

    random_split1, random_split2 = random_split(
        dataset=dataset,
        lengths=[length1, length2],
        generator=torch.manual_seed(42))

    return random_split1, random_split2


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int
):
    """
    Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data. # noqa 5501
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """

    # Create datasets using datasets.ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn datasets into dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names


def main():

    ssl._create_default_https_context = ssl._create_unverified_context # noqa 5501
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])
    train_data, test_data = download_data(data_path=Path("data"),
                                          train_transform=transform,
                                          test_transform=transform)
    train_20, _ = split_data(dataset=train_data, split=0.2)
    print(len(train_data), len(train_20))
    print(len(test_data))
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)
    print(next(iter(train_dataloader))[0].shape)


if __name__ == "__main__":
    main()
