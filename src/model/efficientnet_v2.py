import torch
from lightning import LightningModule
from torch.nn import Linear
from torch.optim import AdamW
from torchmetrics import F1Score, Accuracy, Precision, Recall
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

from .loss.focal_loss import FocalLoss


class EfficientNetV2(LightningModule):
    def __init__(self, datamodule, model_size):
        super().__init__()

        match model_size:
            case 's':
                self.model = efficientnet_v2_s(weights='DEFAULT')
            case 'm':
                self.model = efficientnet_v2_m(weights='DEFAULT')
            case 'l':
                self.model = efficientnet_v2_l(weights='DEFAULT')
            case _:
                raise ValueError(f'Unsupported model_size: {model_size}')

        classifier = self.model.classifier[1]
        self.model.classifier[1] = Linear(in_features=classifier.in_features, out_features=datamodule.num_classes)

        self.loss = FocalLoss(
            alpha=torch.tensor(datamodule.class_weights),
            gamma=2,
            reduction='mean'
        )

        self.train_f1 = F1Score(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.val_f1 = F1Score(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.test_f1 = F1Score(task='multiclass', average='macro', num_classes=datamodule.num_classes)

        self.train_accuracy = Accuracy(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.val_accuracy = Accuracy(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.test_accuracy = Accuracy(task='multiclass', average='macro', num_classes=datamodule.num_classes)

        self.train_precision = Precision(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.val_precision = Precision(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.test_precision = Precision(task='multiclass', average='macro', num_classes=datamodule.num_classes)

        self.train_recall = Recall(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.val_recall = Recall(task='multiclass', average='macro', num_classes=datamodule.num_classes)
        self.test_recall = Recall(task='multiclass', average='macro', num_classes=datamodule.num_classes)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)

    def forward(self, inputs):
        return self.model(inputs)

    def common_step(self, batch):
        pixel_values, class_ids = batch
        logits = self(pixel_values)
        loss = self.loss(logits, class_ids)
        preds = logits.argmax(-1)
        return loss, preds, class_ids

    def training_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        self.train_f1(preds, class_ids)
        self.log('train_f1', self.train_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.train_accuracy(preds, class_ids)
        self.log('train_accuracy', self.train_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.train_precision(preds, class_ids)
        self.log('train_precision', self.train_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.train_recall(preds, class_ids)
        self.log('train_recall', self.train_recall, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        self.val_f1(preds, class_ids)
        self.log('val_f1', self.val_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.val_accuracy(preds, class_ids)
        self.log('val_accuracy', self.val_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.val_precision(preds, class_ids)
        self.log('val_precision', self.val_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.val_recall(preds, class_ids)
        self.log('val_recall', self.val_recall, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('test_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        self.test_f1(preds, class_ids)
        self.log('test_f1', self.test_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.test_accuracy(preds, class_ids)
        self.log('test_accuracy', self.test_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.test_precision(preds, class_ids)
        self.log('test_precision', self.test_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.test_recall(preds, class_ids)
        self.log('test_recall', self.test_recall, prog_bar=False, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        pass
