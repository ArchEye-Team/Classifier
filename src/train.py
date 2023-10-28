from pathlib import Path

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from model.efficientnet_v2 import EfficientNetV2
from data.datamodule import DataModule

if __name__ == '__main__':
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[1]

    data_module = DataModule(
        data_path=root_dir / 'dataset',
        batch_size=5,
        num_workers=4,
        image_size=512
    )

    model = EfficientNetV2(
        data_module,
        model_size='s'
    )

    trainer = Trainer(
        default_root_dir=root_dir,
        deterministic=True,
        max_epochs=-1,
        accelerator='gpu',
        logger=MLFlowLogger(
            tracking_uri='http://212.233.73.128:5000',
            experiment_name='lightning_logs',
            log_model='all'
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'checkpoints',
                save_last=True,
                save_top_k=3,
                monitor='val_f1',
                mode='max',
            ),
            EarlyStopping(
                patience=15,
                monitor='val_f1',
                mode='max',
            )
        ]
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path='last'
    )
