import torchmetrics
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from ExampleDataModule import ExampleDataModule
from argparse import ArgumentParser
import os
from models import ExampleModel

def train(model, name, epochs=40, batch_size=128, debug=False):
    dm = ExampleDataModule('data.csv', './data', batch_size)
    # dm.prepare_data()
    # dm.setup()

    # logger = pl.loggers.TensorBoardLogger("./logs", name, log_graph=True)
    logger = pl.loggers.TensorBoardLogger("./logs", name, version=(None if debug else os.environ["SLURM_JOB_ID"]), log_graph=True)
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=f"./artifacts/{logger.version}", save_top_k=1)

    trainer = Trainer(
        fast_dev_run=False, 
        logger=logger, 
        callbacks=[checkpointer], 
        default_root_dir=f"./artifacts/{logger.version}", 
        accelerator='gpu', 
        devices=1, 
        max_epochs=epochs, 
        deterministic=True, 
        enable_progress_bar=debug,
        # limit_train_batches=100, 
        # limit_val_batches=100, 
        # limit_test_batches=100
        )
    trainer.fit(model, datamodule=dm)

    # Test best model on test set
    trainer.test(model, datamodule=dm, verbose=True)
    
    logger.finalize('success')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='ExampleModel')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    # TRAIN
    pl.utilities.seed.seed_everything(seed=42, workers=True)
    
    N_CLASSES = 2
    
    metrics = {
        'F1': torchmetrics.F1Score(compute_on_step=False),
        'CM': torchmetrics.ConfusionMatrix(num_classes=N_CLASSES, normalize='true')
    }

    model = None
    session_name = args.name if len(args.name) else args.model
    match args.model:
        case 'ExampleModel':
            model = ExampleModel(metrics=metrics, num_classes=N_CLASSES)
    
    train(model, session_name, epochs=args.epochs, debug=args.debug)
