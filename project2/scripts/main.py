
import argparse
import sys
import os
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import Linear, MLP, CNN, ResNet18, CNN3D, ResNet18_3D, ResNet18Video3D, RiskModel
from src.dataset import PathMnist, NLST
from src.gpu_utils import print_gpu_info, configure_ddp_strategy
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl

NAME_TO_MODEL_CLASS = {
    "linear": Linear,
    "mlp": MLP,
    "cnn": CNN,
    "resnet18": ResNet18,
    "cnn3d": CNN3D,
    "resnet18_3d": ResNet18_3D,
    "resnet18_video3d": ResNet18Video3D,
    "risk_model": RiskModel
}

NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="mlp",
        help="Name of model to use. Options include: mlp, cnn, resnet",
    )

    parser.add_argument(
        "--dataset_name",
        default="pathmnist",
        help="Name of dataset to use. Options: pathmnist, nlst"
    )

    parser.add_argument(
        "--project_name",
        default="cornerstone",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_loss",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether to train the model."
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    import os

    # Set CUDA memory management to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print(args)
    print("Loading data ..")

    # Print GPU information and check NVLink
    print_gpu_info()

    # Configure DDP strategy based on NVLink availability
    # Will use the devices specified in args.trainer.devices, or auto-detect
    requested_devices = getattr(args.trainer, 'devices', None)
    if requested_devices == 'auto' or requested_devices is None:
        requested_devices = None  # Auto-detect

    strategy, num_devices = configure_ddp_strategy(devices=requested_devices)

    print("Preparing lighning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """
    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**vars(args[args.dataset_name]))

    print("Initializing model")
    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    # Enable WandB logger
    logger = pl.loggers.WandbLogger(entity="curtis-chambers-university-of-california-berkeley",project=args.project_name, offline=False)

    args.trainer.accelerator = 'auto'
    args.trainer.devices = num_devices
    args.trainer.strategy = strategy
    args.trainer.logger = logger
    # REMOVED: hardcoded accumulate_grad_batches - now configurable per training script
    # The old hardcoded value of 8 was causing NaN issues with batch_size=2
    if not hasattr(args.trainer, 'accumulate_grad_batches') or args.trainer.accumulate_grad_batches is None:
        args.trainer.accumulate_grad_batches = 1  # Default to no accumulation
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor=args.monitor_key,  # Watch validation metric
            patience=5,  # Stop after 5 epochs without improvement
            mode='min' if "loss" in args.monitor_key else "max",
            verbose=True,  # Print when stopping
            min_delta=0.001  # Minimum change to qualify as improvement
        )]

    trainer = pl.Trainer(**vars(args.trainer))

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)
        print("Training completed")
        
        # Get best checkpoint path if training was done
        if trainer.checkpoint_callback:
            print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)
    print("Validation completed")

    print("Evaluating model on test set")
    trainer.test(model, datamodule)
    print("Test completed")

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
