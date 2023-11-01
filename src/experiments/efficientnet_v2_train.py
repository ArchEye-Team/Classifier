from pathlib import Path

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from src.data.datamodule import DataModule
from src.model.efficientnet_v2 import EfficientNetV2

if __name__ == '__main__':
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[2]

    datamodule = DataModule(
        data_path=root_dir / 'dataset_cleaned',
        batch_size=4,
        num_workers=4
    )

    model = EfficientNetV2(
        datamodule=datamodule,
        model_size='s'
    )

    trainer = Trainer(
        default_root_dir=root_dir,
        deterministic=True,
        max_epochs=-1,
        accelerator='gpu',
        logger=MLFlowLogger(experiment_name='EfficientNetV2'),
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'checkpoints',
                filename='{epoch}-{val_f1:.2f}',
                monitor='val_f1',
                mode='max',
                save_last=True,
                save_top_k=3,
            ),
            EarlyStopping(
                patience=10,
                monitor='val_f1',
                mode='max',
            )
        ]
    )

    try:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path='last'
        )
    finally:
        print('Training interrupted, start testing')
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path='best'
        )
