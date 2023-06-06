import torch
from pathlib import Path
import argparse
from typing import List
import torchvision
from tqdm.auto import tqdm
from timeit import default_timer as timer
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from typing import Tuple


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    torch.save(obj=model.state_dict(),
               f=model_save_path)


def set_seeds(seed: int = 42):

    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def parse_arguments():
    """
    Parse arguments:
    - batch size
    - hidden units
    - learning rate as lr
    - num of epochs
    """
    parser = argparse.ArgumentParser(description='Script train model TinyVGG')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--hidden_units', type=int, default=10,
                        help='Number of hidden units in the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs for training')

    args = parser.parse_args()
    return args


def pred_and_store(paths: List[Path],
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device="cuda"):

    pred_list = []
    for path in tqdm(paths):
        pred_dict = {}
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        start_time = timer()

        img = Image.open(path)

        transformed_img = transform(img).unsqueeze(0).to(device)

        model.to(device)
        model.eval()

        with torch.inference_mode():
            pred_logit = model(transformed_img)
            pred_prob = torch.softmax(input=pred_logit, dim=1)
            pred_label = torch.argmax(input=pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            pred_dict["pred_prob"] = pred_label.item()
            pred_dict["pred_class"] = pred_class

            end_time = timer()
            pred_dict["prediction_time"] = round(end_time - start_time, 4)

        pred_dict["correct"] = class_name == pred_class

        pred_list.append(pred_dict)

    return pred_list


def predict(img: Image,
            model: torch.nn.Module,
            transform: torchvision.transforms,
            class_names: List[str],
            device="cuda"):

    start_time = timer()

    transformed_img = transform(img).unsqueeze(0).to(device)

    model.to(device)
    model.eval()

    with torch.inference_mode():
        pred_logit = model(transformed_img)
        pred_prob = torch.softmax(input=pred_logit, dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))} # noqa 5501

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> SummaryWriter:
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir. # noqa 5501
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra. # noqa 5501
    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Nome of experiment.
        model_name (str): Name of the model.
        extra (str, optional): Anything extra to add int dir . Defaults to None.

    Returns:
        SummaryWriter: instance of a writer
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra) # noqa 5501
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def pred_and_plot_image(model: nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = "cuda"):
    # Open an image
    img = Image.open(image_path)

    # Create transformation
    if transform:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        image_pred = model(transformed_image.to(device))

    img_pred_probs = torch.softmax(input=image_pred, dim=1)
    img_label = torch.argmax(input=img_pred_probs, dim=1)
    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[img_label]} | Prob: {img_pred_probs.max():.3f}') # noqa 5501
    plt.axis(False)
    plt.show()
