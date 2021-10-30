import sklearn.metrics as metrics
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser, Namespace

from utils.dotdict import dotdict
import pytorch_lightning.metrics.functional as F_m
import wandb
import pandas as pd

from models.audio_modal.datasets_loaders.iemocap_spect import IEMOCAPSpectDataset


class Davenet(nn.Module):
    # based on https://github.com/dharwath/DAVEnet-pytorch
    def __init__(
        self,
        embedding_dim=1024,
        dropout_1=0.0,
        dropout_2=0.0,
        dropout_3=0.0,
        dropout_4=0.0,
    ):
        super(Davenet, self).__init__()

        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(3)
        conv1_f = int(embedding_dim / 8)
        conv2_f = int(embedding_dim / 4)
        conv3_f = int(embedding_dim / 2)

        self.conv1 = nn.Conv2d(
            3, conv1_f, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0)
        )

        self.conv2 = nn.Conv2d(
            conv1_f, conv2_f, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5)
        )
        self.conv3 = nn.Conv2d(
            conv2_f, conv3_f, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8)
        )
        self.conv4 = nn.Conv2d(
            conv3_f, conv3_f, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8)
        )

        self.conv5 = nn.Conv2d(
            conv3_f, embedding_dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.dropout_1 = nn.Dropout2d(p=dropout_1)
        self.dropout_2 = nn.Dropout2d(p=dropout_2)
        self.dropout_3 = nn.Dropout2d(p=dropout_3)
        self.dropout_4 = nn.Dropout2d(p=dropout_4)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = self.dropout_1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout_3(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout_4(x)

        x = F.relu(self.conv5(x))
        return x


def permute_audio(x):
    return x.permute(1, 2, 0)


class DaveAudionet(pl.LightningModule):
    def __init__(self, hparams):
        super(DaveAudionet, self).__init__()
        if not isinstance(hparams, Namespace):
            hparams = dotdict(hparams)
        self.save_hyperparameters(hparams)
        self.num_classes = self.hparams.num_classes
        self.input_dim = self.hparams.input_dim
        self.n_hidden = self.hparams.n_hidden
        self.audio_emb = self.hparams.audio_emb
        print("audio hparams")
        print(self.hparams)

        # self.example_input_array = torch.zeros(1, 3, 40, 300)
        if hasattr(self.hparams, "dropout_1"):
            self.audio_dave = Davenet(
                self.input_dim,
                self.hparams.dropout_1,
                self.hparams.dropout_2,
                self.hparams.dropout_3,
                self.hparams.dropout_4,
            )
        else:
            self.audio_dave = Davenet(self.input_dim)

        self.cnn1 = nn.Conv2d(self.input_dim, self.input_dim, (1, 10), padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.dropout = nn.Dropout2d(p=self.hparams.dropout_audio)

        self.fc1 = nn.Linear(5120, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x):
        x = self.audio_dave(x)
        x = self.max_pool(x)
        x = F.relu(self.cnn1(x))
        x = self.dropout(x)
        if self.audio_emb == 0:
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        y = y.detach()
        y_hat = y_hat.detach()
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        acc = F_m.classification.precision(y_pred, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
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

        acc_micro = F_m.classification.precision(y_pred, y, class_reduction="micro")
        acc_macro = F_m.classification.precision(y_pred, y, class_reduction="macro")
        acc_weighted = F_m.classification.precision(
            y_pred, y, class_reduction="weighted"
        )
        f1 = F_m.f1(y_pred, y, num_classes=self.num_classes)
        return {
            "val_loss": loss_val,
            "val_f1": f1,
            "val_acc_micro": acc_micro,
            "val_acc_macro": acc_macro,
            "val_acc_weighted": acc_weighted,
        }

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        for metric_name in [
            "val_loss",
            "val_f1",
            "val_acc_micro",
            "val_acc_macro",
            "val_acc_weighted",
        ]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]
                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)
        if "best_val_loss" in self.logger.experiment.summary.keys():
            if (
                tqdm_dict["val_loss"].cpu().numpy()
                < self.logger.experiment.summary["best_val_loss"]
            ):
                self.logger.experiment.summary["best_val_loss"] = (
                    tqdm_dict["val_loss"].cpu().numpy()
                )
        else:
            self.logger.experiment.summary["best_val_loss"] = (
                tqdm_dict["val_loss"].cpu().numpy()
            )

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat, y

    def test_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]

        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]

        acc_micro = F_m.classification.precision(y_pred, y, class_reduction="micro")
        acc_macro = F_m.classification.precision(y_pred, y, class_reduction="macro")
        acc_weighted = F_m.classification.precision(
            y_pred, y, class_reduction="weighted"
        )

        f1_micro = F_m.f1(y_pred, y, num_classes=self.num_classes, average="micro")
        f1_macro = F_m.f1(y_pred, y, num_classes=self.num_classes, average="macro")
        f1_weighted = F_m.f1(
            y_pred, y, num_classes=self.num_classes, average="weighted"
        )

        return {
            "test_loss": loss_val,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "test_acc_micro": acc_micro,
            "test_acc_macro": acc_macro,
            "test_acc_weighted": acc_weighted,
            "res": (y_pred.cpu(), y.cpu()),
        }

    def test_epoch_end(self, outputs):
        # OPTIONAL
        tqdm_dict = {}
        for metric_name in [
            "test_loss",
            "test_f1_micro",
            "test_f1_macro",
            "test_f1_weighted",
            "test_acc_micro",
            "test_acc_macro",
            "test_acc_weighted",
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
                    F_m.confusion_matrix(
                        y_pred.type(torch.ByteTensor),
                        y.type(torch.ByteTensor),
                        num_classes=self.num_classes,
                    )
                    .cpu()
                    .numpy()
                )
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

    def train_dataloader(self):
        if self.hparams.dataset == "iemocap":
            transform = transforms.Compose([transforms.ToTensor(), permute_audio,])
            dataset = IEMOCAPSpectDataset(
                self.hparams.audio_folder,
                set_type="train",
                transform=transform,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        # OPTIONAL

        if self.hparams.dataset == "iemocap":
            transform = transforms.Compose([transforms.ToTensor(), permute_audio,])
            dataset = IEMOCAPSpectDataset(
                self.hparams.audio_folder,
                set_type="val",
                transform=transform,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.hparams.dataset == "iemocap":
            transform = transforms.Compose([transforms.ToTensor(), permute_audio,])
            dataset = IEMOCAPSpectDataset(
                self.hparams.audio_folder,
                set_type="test",
                transform=transform,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--weight_decay", default=0.01, type=float)

        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--input_dim", default=512, type=int)
        parser.add_argument("--n_hidden", default=438, type=int)
        parser.add_argument("--dropout_audio", default=0.3995, type=float)
        parser.add_argument("--dropout_1", default=0.2617, type=float)
        parser.add_argument("--dropout_2", default=0.1342, type=float)
        parser.add_argument("--dropout_3", default=0.2902, type=float)
        parser.add_argument("--dropout_4", default=0.3718, type=float)

        parser.add_argument("--audio_emb", default=0, type=int)

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=10000, type=int)

        # data
        parser.add_argument("--num_classes", dest="num_classes", default=4, type=int)
        return parser
