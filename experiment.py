from data_setup import create_dataloaders
import utils
from model_builder import create_effnetb2, create_resnet50, create_effnetb0
import engine

import torch
from torch import nn

import os
from pathlib import Path
import mlflow


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = os.cpu_count()
    num_classes = 3
    mlflow.set_experiment("Pet Recognition")

    # # Create models
    effnetb2, effnetb2_transforms = create_effnetb2(
        num_of_classes=num_classes, device=device
    )
    resnet50, resnet50_transforms = create_resnet50(
        num_of_classes=num_classes, device=device
    )
    effnetb0, effnetb0_transforms = create_effnetb0(
        out_features=num_classes, device=device
    )

    # # Create dataloaders
    data_path = Path("data_3")
    train_path = data_path / "train"
    test_path = data_path / "test"
    (
        train_dataloader_effnetb2,
        test_dataloader_effnetb2,
        class_names,
    ) = create_dataloaders(
        train_dir=train_path,
        test_dir=test_path,
        transform=effnetb2_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    (
        train_dataloader_effnetb0,
        test_dataloader_effnetb0,
        class_names,
    ) = create_dataloaders(
        train_dir=train_path,
        test_dir=test_path,
        transform=effnetb0_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    (
        train_dataloader_resnet50,
        test_dataloader_resnet50,
        class_names,
    ) = create_dataloaders(
        train_dir=train_path,
        test_dir=test_path,
        transform=resnet50_transforms,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # # Create experiment
    epochs = [5, 10]
    models = ["resnet50", "effnetb0", "effnetb2"]
    experiment_num = 0

    # # Experiment
    for epoch in epochs:
        for model_name in models:
            experiment_num += 1
            print(f"[INFO] Experiment number: {experiment_num}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] Number of epochs: {epoch}")

            # Create model
            if model_name == "effnetb2":
                model, _ = create_effnetb2(num_of_classes=num_classes, device=device)
                train_dataloader = train_dataloader_effnetb2
                test_dataloader = test_dataloader_effnetb2
            elif model_name == "effnetb0":
                model, _ = create_effnetb0(out_features=num_classes, device=device)
                train_dataloader = train_dataloader_effnetb0
                test_dataloader = test_dataloader_effnetb0
            else:
                model, _ = create_resnet50(num_of_classes=num_classes, device=device)
                train_dataloader = train_dataloader_resnet50
                test_dataloader = test_dataloader_resnet50

            # Setup loss fn and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            with mlflow.start_run(run_name=f"{model_name}"):
                mlflow.log_param("epochs", epoch)

                results = engine.train(
                    model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epoch,
                    writer=utils.create_writer(
                        experiment_name=f"{epoch} Epochs", model_name=model_name
                    ),
                    device=device,
                )
                for key, value in results.items():
                    for idx, val in enumerate(value):
                        mlflow.log_metric(f"{key}_{idx}", val)

                # Save model
                save_name = f"Pretrained_{model_name}_{epoch}_epochs.pth"
                utils.save_model(model=model, target_dir="models", model_name=save_name)
                mlflow.pytorch.log_model(model, "models")

            print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
