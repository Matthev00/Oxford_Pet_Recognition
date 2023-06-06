from data_setup import download_data, split_data, create_dataloaders
import utils
import model_builder
import engine

import torch # noqa 5501
import torchvision # noqa 5501
from torch import nn # noqa 5501
from torchvision import transforms, datasets # noqa 5501data_20_percent_path
from torchinfo import summary # noqa 5501

import os
from pathlib import Path
import random


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

    # # # Train model
    # optimizer = torch.optim.Adam(effnetb2.parameters(), lr=1e-2)
    # loss_fn = nn.CrossEntropyLoss()
    # result = engine.train(model=effnetb2,
    #                       train_dataloader=train_dataloader,
    #                       test_dataloader=test_dataloader,
    #                       optimizer=optimizer,
    #                       loss_fn=loss_fn,
    #                       epochs=EPOCHS,
    #                       writer=utils.create_writer(experiment_name="test",
    #                                                  model_name="xyz"),
    #                       device=device)

    # utils.plot_loss_curves(results=result)

    # # Load best model
    best_model_path = "models/Pretrained_resnet50_10_epochs.pth"
    resnet50.load_state_dict(torch.load(best_model_path))

    model_size = Path(best_model_path).stat().st_size // (1024*1024)
    print(f"Model size {model_size}")

    # # Predict on image
    img_to_plot = 5
    test_image_path_list = list(Path(test_path).glob("*/*.jpg"))
    img_path_sample = random.sample(population=test_image_path_list,
                                    k=img_to_plot)
    for img_path in img_path_sample:
        utils.pred_and_plot_image(model=resnet50,
                                  image_path=img_path,
                                  class_names=class_names,
                                  image_size=(288, 288))


if __name__ == "__main__":
    main()
