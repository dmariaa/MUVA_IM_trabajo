import click
from pytorch_lightning import Trainer

from dataset import GlassData
from lightning_model import load_checkpoint


@click.command()
@click.option("--check-point", help="Checkpoint to load", required=True)
@click.option("-bs", "--batch-size", help="Batch size", required=True, type=int)
@click.option("--gpu", is_flag=True)
def test(check_point, batch_size, gpu):
    model = load_checkpoint(check_point, gpu, version=3)

    datamodule = GlassData(data_folder='augmented_dataset', batch_size=batch_size, validation_split=0.1, num_workers=4)
    if gpu:
        trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1, logger=None)
    else:
        trainer = Trainer(max_epochs=1, logger=None)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()
