import os.path

import click
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from lightning_model import load_checkpoint


@click.command()
@click.argument("file", type=str)
@click.option("--check-point", help="Checkpoint to load", required=True)
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option("--model-version", help="Model version", type=int, default=3)
def predict(file, check_point, gpu, model_version):
    model = load_checkpoint(check_point, gpu, version=model_version)
    model.eval()

    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t_image = torch.unsqueeze(transforms.ToTensor()(image), 0)

    fname, ext = os.path.splitext(file)
    gt = cv2.imread(f"{fname}_anno{ext}")
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt[gt > 0] = 1
    gt = gt.astype(np.uint8)[..., 0]

    with torch.no_grad():
        pred = model(t_image)

    pred = torch.squeeze(pred).cpu().detach().numpy()
    pred = (pred > 0.5).astype(np.uint8)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9.6, 4.8))
    ax0.imshow(image)
    ax1.imshow(image)
    ax1.imshow(80 * gt, alpha=0.7)
    ax2.imshow(image)
    ax2.imshow(80 * pred, alpha=0.7)
    fig.show()
    plt.show()


if __name__ == "__main__":
    predict()
