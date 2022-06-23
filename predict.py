import click
from lightning_model import load_checkpoint


@click.command()
@click.argument("file", type=str)
@click.option("--check-point", help="Checkpoint to load", required=True)
@click.option("--gpu", is_flag=True)
def predict(check_point, gpu):
    model = load_checkpoint(check_point, gpu, version=3)


if __name__ == "__main__":
    predict()
