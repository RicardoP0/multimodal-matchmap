# %%
import sklearn.metrics as metrics
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser, Namespace
import wandb
import pandas as pd

import torchvision.transforms._transforms_video as video_transforms
from models.video_modal.data_aug.temporal_transforms import TemporalEvenSample
from models.audio_modal.datasets_loaders.iemocap_spect import permute_audio

from models.multimodal.dataset_loader.iemocapmultimodal import IEMOCAPMultiModalDataset
from utils.dotdict import dotdict
from models.video_modal.resnet3d import ResNet3dVideo
from models.audio_modal.davenet_audio import DaveAudionet
from torchmetrics.functional.classification.f_beta import f1
import torchmetrics


class MultiModalConcatNet(pl.LightningModule):
    def __init__(self, hparams):
        super(MultiModalConcatNet, self).__init__()
        if not isinstance(hparams, Namespace):
            hparams = dotdict(hparams)
        self.save_hyperparameters(hparams)
        self.num_classes = self.hparams.num_classes
        self.videonet = ResNet3dVideo.load_from_checkpoint(
            self.hparams.video_pretrained_folder
            + "/Resnet3dVideoFold-"
            + str(self.hparams.fold)
            + ".ckpt",
            video_emb=1,
            strict=False,
        )
        self.audio_dave = DaveAudionet.load_from_checkpoint(
            self.hparams.audio_pretrained_folder
            + "/AudioModalFold-"
            + str(self.hparams.fold)
            + ".ckpt",
            audio_emb=1,
        )

        if self.hparams.freeze_video:
            self.videonet.freeze()

        if self.hparams.freeze_audio:
            self.audio_dave.freeze()

        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x_audio = x[0]
        x_video = x[1]
        x_audio = self.audio_dave(x_audio)
        x_audio = F.max_pool2d(x_audio, (x_audio.shape[2], x_audio.shape[3]))
        x_audio = x_audio.squeeze(2).squeeze(2).flatten(1)

        # video
        x_video = self.videonet(x_video)
        x_video = F.max_pool3d(
            x_video, (x_video.shape[2], x_video.shape[3], x_video.shape[4])
        )
        x_video = x_video.squeeze(2).flatten(1)

        x = torch.cat((x_audio, x_video), dim=1)
        if self.hparams.multimodal_emb == 0:
            x = self.fc(x)
        return x, (x_audio, x_video)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x_audio, x_video, y = batch["audio"], batch["video"], batch["label"]
        y_hat, x_emb = self.forward((x_audio, x_video))
        loss = F.cross_entropy(y_hat, y)

        y = y.detach()
        y_hat = y_hat.detach()
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        acc = torchmetrics.functional.accuracy(y_pred, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_audio, x_video, y = batch["audio"], batch["video"], batch["label"]
        y_hat, _ = self.forward((x_audio, x_video))
        return y_hat, y

    def validation_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        acc = torchmetrics.functional.accuracy(y_pred, y)
        prec_micro = torchmetrics.functional.precision(
            y_pred, y, average="micro", num_classes=self.num_classes
        )
        prec_macro = torchmetrics.functional.precision(
            y_pred, y, average="macro", num_classes=self.num_classes
        )
        prec_weighted = torchmetrics.functional.precision(
            y_pred, y, average="weighted", num_classes=self.num_classes
        )
        f1_val = f1(y_pred, y, num_classes=self.num_classes, average="macro")
        return {
            "val_loss": loss_val,
            "val_f1": f1_val,
            "val_acc": acc,
            "val_prec_micro": prec_micro,
            "val_prec_macro": prec_macro,
            "val_prec_weighted": prec_weighted,
        }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        tqdm_dict = {}

        for metric_name in [
            "val_loss",
            "val_f1",
            "val_acc",
            "val_prec_micro",
            "val_prec_macro",
            "val_prec_weighted",
        ]:

            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x_audio, x_video, y = batch["audio"], batch["video"], batch["label"]
        y_hat, _ = self.forward((x_audio, x_video))
        return y_hat, y

    def test_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        # do softmax here
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]

        acc = torchmetrics.functional.accuracy(y_pred, y)

        prec_micro = torchmetrics.functional.precision(
            y_pred, y, average="micro", num_classes=self.num_classes
        )
        prec_macro = torchmetrics.functional.precision(
            y_pred, y, average="macro", num_classes=self.num_classes
        )
        prec_weighted = torchmetrics.functional.precision(
            y_pred, y, average="weighted", num_classes=self.num_classes
        )

        rec_micro = torchmetrics.functional.recall(
            y_pred, y, average="micro", num_classes=self.num_classes
        )
        rec_macro = torchmetrics.functional.recall(
            y_pred, y, average="macro", num_classes=self.num_classes
        )
        rec_weighted = torchmetrics.functional.recall(
            y_pred, y, average="weighted", num_classes=self.num_classes
        )

        f1_micro = f1(y_pred, y, num_classes=self.num_classes, average="micro")
        f1_macro = f1(y_pred, y, num_classes=self.num_classes, average="macro")
        f1_weighted = f1(y_pred, y, num_classes=self.num_classes, average="weighted")

        return {
            "test_loss": loss_val,
            "test_acc": acc,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "test_prec_micro": prec_micro,
            "test_prec_macro": prec_macro,
            "test_prec_weighted": prec_weighted,
            "test_rec_micro": rec_micro,
            "test_rec_macro": rec_macro,
            "test_rec_weighted": rec_weighted,
            "res": (y_pred.cpu(), y.cpu()),
        }

    def test_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in [
            "test_loss",
            "test_acc",
            "test_f1_micro",
            "test_f1_macro",
            "test_f1_weighted",
            "test_prec_micro",
            "test_prec_macro",
            "test_prec_weighted",
            "test_rec_micro",
            "test_rec_macro",
            "test_rec_weighted",
            "res",
        ]:
            if metric_name == "res":
                y_pred = torch.zeros((1))
                y = torch.zeros((1))

                for output in outputs:
                    n_ypred, n_y = output[metric_name]
                    y_pred = torch.cat((y_pred, n_ypred))
                    y = torch.cat((y, n_y))
                conf_matrix = (
                    torchmetrics.functional.classification.confusion_matrix(
                        y_pred.type(torch.ByteTensor),
                        y.type(torch.ByteTensor),
                        num_classes=self.num_classes,
                    )
                    .cpu()
                    .numpy()
                )
                df = pd.DataFrame(data=y_pred, columns=["pred"])
                df["true"] = y

                report = metrics.classification_report(
                    y.type(torch.ByteTensor),
                    y_pred.type(torch.ByteTensor),
                    output_dict=True,
                )
            else:
                metric_total = 0
                for output in outputs:
                    metric_value = output[metric_name]

                    # reduce manually when using dp
                    if self.trainer.use_dp or self.trainer.use_ddp2:
                        metric_value = metric_value.mean()

                    metric_total += metric_value

                tqdm_dict[metric_name] = metric_total / len(outputs)
        unique_label = list(range(self.num_classes))
        print(unique_label)
        print(conf_matrix)
        # print(df_pred)
        cmtx = pd.DataFrame(
            conf_matrix,
            index=["true:{:}".format(x) for x in unique_label],
            columns=["pred:{:}".format(x) for x in unique_label],
        )
        report_log = pd.DataFrame(report)
        print(report)

        dict_a = {
            "confusion_matrix": wandb.plots.HeatMap(
                unique_label, unique_label, cmtx.values, show_text=True
            ),
            "classification_report": wandb.Table(dataframe=report_log),
            "outputs": wandb.Table(dataframe=df),
        }
        self.logger.experiment.log(dict_a)

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_data(self) -> None:

        video_transform = transforms.Compose([video_transforms.ToTensorVideo(),])
        vtemptr = transforms.Compose([TemporalEvenSample(16)])
        audio_transform = transforms.Compose([transforms.ToTensor(), permute_audio,])

        self.train_dataset = IEMOCAPMultiModalDataset(
            self.hparams.audio_folder,
            self.hparams.video_folder,
            set_type="train",
            video_transform=video_transform,
            vtemporal_transform=vtemptr,
            audio_transform=audio_transform,
            num_classes=self.num_classes,
            fold=self.hparams.fold,
        )
        self.val_dataset = IEMOCAPMultiModalDataset(
            self.hparams.audio_folder,
            self.hparams.video_folder,
            set_type="val",
            video_transform=video_transform,
            vtemporal_transform=vtemptr,
            audio_transform=audio_transform,
            num_classes=self.num_classes,
            fold=self.hparams.fold,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            timeout=30,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            timeout=300,
        )

    def test_dataloader(self):
        video_transform = transforms.Compose([video_transforms.ToTensorVideo(),])
        vtemptr = transforms.Compose([TemporalEvenSample(16)])
        audio_transform = transforms.Compose([transforms.ToTensor(), permute_audio,])

        dataset = IEMOCAPMultiModalDataset(
            self.hparams.audio_folder,
            self.hparams.video_folder,
            set_type="test",
            video_transform=video_transform,
            vtemporal_transform=vtemptr,
            audio_transform=audio_transform,
            num_classes=self.num_classes,
            fold=self.hparams.fold,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            timeout=30,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # OPTIMIZER ARGS
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--weight_decay", default=0.01, type=float)

        # MULTIMODAL
        parser.add_argument("--multimodal_emb", default=0, type=int)
        parser.add_argument("--freeze_audio", default=0, type=int)
        parser.add_argument("--freeze_video", default=0, type=int)

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=10000, type=int)

        parser.add_argument("--num_classes", dest="num_classes", default=4, type=int)

        return parser

