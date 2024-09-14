import gradio as gr
import os
import torch

from model import create_effnetb2
from timeit import default_timer as timer
from PIL import Image


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Setup class names
    with open("class_names.txt", "r") as filehandle:
        class_names = [bread.strip() for bread in filehandle.readlines()]

    # # Create model
    model, model_transforms = create_effnetb2(
        num_of_classes=len(class_names), device=device
    )

    model.load_state_dict(
        torch.load(f="Pretrianed_effnetb2.pth", map_location=torch.device(device))
    )

    def predict(img: Image):
        start = timer()

        transformed_img = model_transforms(img).unsqueeze(0).to(device)

        model.to(device)
        model.eval()

        with torch.inference_mode():
            pred_logit = model(transformed_img)
            pred_prob = torch.softmax(input=pred_logit, dim=1)

        pred_labels_and_probs = {
            class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))
        }

        pred_time = round(timer() - start, 5)

        return pred_labels_and_probs, pred_time

    title = "Pet recognition  🐶🐱"
    description = "An EfficientNetB2 feature extractor computer vision model to classify images of pets."
    example_list = [
        ["examples/" + example] for example in os.listdir("examples")
    ]

    # # Create Gradio
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=3, label="Predictions"),
            gr.Number(label="Prediction time (s)"),
        ],
        examples=example_list,
        title=title,
        description=description,
    )

    demo.launch()


if __name__ == "__main__":
    main()
