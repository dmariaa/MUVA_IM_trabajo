import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import GlassData
from lightning_model import LightningModel
from model_unet_v3 import Unet


@click.command()
@click.option("-bs", "--batch-size", help="Batch size", required=True, type=int)
@click.option("-e", "--epochs", help="Number of epochs", required=True, type=int)
@click.option("-lr", "--learning-rate", help="Learning rate", required=True, type=float)
@click.option("-nw", "--num-workers", help="Number of workers", default=4, type=int)
@click.option("--gpu", is_flag=True)
def train(batch_size, epochs, learning_rate, num_workers, gpu):
    model = Unet(input_channels=3, use_drop_out=True)
    lm = LightningModel(model, learning_rate=learning_rate)
    datamodule = GlassData(data_folder='augmented_dataset', batch_size=batch_size, validation_split=0.1,
                           num_workers=num_workers)

    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="val_loss")
    if gpu:
        trainer = Trainer(max_epochs=epochs, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])

    trainer.fit(model=lm, datamodule=datamodule)


if __name__ == "__main__":
    train()
