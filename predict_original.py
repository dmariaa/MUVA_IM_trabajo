import os.path

import click
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from dataset_generator import normalize_color
from lightning_model import load_checkpoint
from smooth_tiled_predictions import predict_img_with_smooth_windowing


def make_prediction(image, model):
    t_image = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2) / 255.

    with torch.no_grad():
        pred = model(t_image)

    pred = pred.cpu().detach().permute(0, 2, 3, 1).numpy()
    pred[pred <= 0.5] = 0.
    pred[pred > 0.5] = 1.
    return pred


@click.command()
@click.argument("file", type=str)
@click.option("--check-point", help="Checkpoint to load", required=True)
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option("--model-version", help="Model version", type=int, default=3)
def predict(file, check_point, gpu, model_version):
    model = load_checkpoint(check_point, gpu, version=model_version)
    model.eval()

    reference_image = cv2.imread("dataset/testA_1.bmp")
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_image = cv2.medianBlur(reference_image, 3)

    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.medianBlur(image, 3)
    image = normalize_color(image, reference_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred = predict_img_with_smooth_windowing(
        input_img=image,
        window_size=256,
        subdivisions=2,
        nb_classes=1,
        pred_func=lambda img_batch_subdiv: make_prediction(img_batch_subdiv, model)
    )

    fname, ext = os.path.splitext(file)
    gt = cv2.imread(f"{fname}_anno{ext}")
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt[gt > 0] = 1
    gt = gt.astype(np.uint8)[..., 0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9.6, 4.8), tight_layout={
        'rect': [0, 0, 1, 0.98]
    })
    ax0.imshow(image)
    ax1.imshow(image)
    ax1.imshow(80 * gt, alpha=0.7)
    ax2.imshow(image)
    ax2.imshow(80 * pred, alpha=0.7)
    fig.show()
    plt.show()


if __name__ == "__main__":
    predict()
