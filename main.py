from data_setup import download_data, split_data, create_dataloaders
import utils
import model_builder
import engine

import torch # noqa 5501
import torchvision # noqa 5501
from torch import nn # noqa 5501
from torchvision import transforms, datasets # noqa 5501data_20_percent_path
from torchinfo import summary # noqa 5501
from torchmetrics import ConfusionMatrix # noqa 5501
from torch.utils.data import DataLoader # noqa 5501
from torch.utils.tensorboard import SummaryWriter # noqa 5501

import os
import random # noqa 5501
from pathlib import Path # noqa 5501


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = os.cpu_count()

    # # Create models
    effnetb2, effnetb2_transforms = model_builder.create_effnetb2(
        num_of_classes=3,
        device=device)
    resnet50, resnet50_transforms = model_builder.create_resnet50(
        num_of_classes=3,
        device=device)
    train_transform = transforms.Compose(
        [transforms.TrivialAugmentWide(),
         effnetb2_transforms])

    # # # Create data (all classes)
    # train_data, test_data = download_data(data_path=Path("data"),
    #                                       train_transform=train_transform,
    #                                       test_transform=effnetb2_transforms)
    # class_names = train_data.classes

    # train_data_20, _ = split_data(dataset=train_data, split=0.2)
    # test_data_20, _ = split_data(dataset=test_data, split=0.2)

    # # 3 classes
    data_path = Path("data_3")
    train_path = data_path / "train"
    test_path = data_path / "test"
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_path,
        test_dir=test_path,
        transform=effnetb2_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)
    # train_data = torchvision.datasets.ImageFolder(root="data_3/train",
    #                                               transform=train_transform)
    # test_data = torchvision.datasets.ImageFolder(root="data_3/test",
    #                                              target_transform=effnetb2)
    # print(train_data.classes)

    # train_dataloader = DataLoader(dataset=train_data,
    #                               batch_size=BATCH_SIZE,
    #                               shuffle=True,
    #                               pin_memory=True,
    #                               num_workers=NUM_WORKERS)

    # test_dataloader = DataLoader(dataset=test_data,
    #                              batch_size=BATCH_SIZE,
    #                              shuffle=False,
    #                              pin_memory=True,
    #                              num_workers=NUM_WORKERS)

    # # Train model
    optimizer = torch.optim.Adam(effnetb2.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    result = engine.train(model=effnetb2,
                          train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=EPOCHS,
                          writer=utils.create_writer(experiment_name="test",
                                                     model_name="xyz"),
                          device=device)

    utils.plot_loss_curves(results=result)



if __name__ == "__main__":
    main()
