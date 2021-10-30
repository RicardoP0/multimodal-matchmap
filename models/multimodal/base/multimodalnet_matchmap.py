import sklearn.metrics as metrics
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
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


def batch_computeMatchmap(I, A):
    assert I.dim() == 4
    assert A.dim() == 3
    D = I.size(1)
    H = I.size(2)
    W = I.size(3)
    T = A.size(2)
    # Ir = I.view(I.size(0),D, -1).permute(0,2,1)
    # matchmap = torch.bmm(Ir, A)
    # matchmap = matchmap.view(I.size(0), H, W, T)
    matchmap = torch.einsum("b c i j, b c t -> b i j t", I, A)
    return matchmap


def batch_matchmapSim(M, simtype):
    assert M.dim() == 4
    if simtype == "SISA":
        return M.mean(dim=(1, 2, 3))
    elif simtype == "MISA":
        M_maxH, _ = M.max(1)
        # print(M_maxH.shape)
        M_maxHW, _ = M_maxH.max(1)
        # print(M_maxHW.shape)

        return M_maxHW.mean(dim=(1))
    elif simtype == "SIMA":
        M_maxT, _ = M.max(3)
        return M_maxT.mean(dim=(1, 2))
    else:
        raise ValueError


def batch_sampled_margin_rank_loss(
    image_outputs, audio_outputs, labels, margin=1.0, simtype="MISA"
):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    based on https://github.com/dharwath/DAVEnet-pytorch
    """
    assert image_outputs.dim() == 4
    assert audio_outputs.dim() == 3
    n = image_outputs.size(0)
    I_imp_ind = torch.randperm(n)
    A_imp_ind = torch.randperm(n)
    for i in range(n):
        # get video with different labels
        if labels[i] == labels[I_imp_ind[i]]:
            imp_label = labels[I_imp_ind[i]]
            n_labels = torch.nonzero(labels != imp_label)
            if n_labels.size(0) > 0:
                idx = n_labels[torch.randint(n_labels.size(0), (1,))].squeeze()
                I_imp_ind[i] = idx
        # get audio with different labels
        if labels[i] == labels[A_imp_ind[i]]:
            imp_label = labels[A_imp_ind[i]]
            n_labels = torch.nonzero(labels != imp_label)
            if n_labels.size(0) > 0:
                idx = n_labels[torch.randint(n_labels.size(0), (1,))].squeeze()
                A_imp_ind[i] = idx

    anchorsim = batch_matchmapSim(
        batch_computeMatchmap(image_outputs, audio_outputs), simtype
    )
    Iimpsim = batch_matchmapSim(
        batch_computeMatchmap(image_outputs[I_imp_ind], audio_outputs), simtype
    )
    Aimpsim = batch_matchmapSim(
        batch_computeMatchmap(image_outputs, audio_outputs[A_imp_ind]), simtype
    )

    A2I_simdif = torch.clamp(margin + Iimpsim - anchorsim, min=0.0)
    loss = A2I_simdif
    I2A_simdif = torch.clamp(margin + Aimpsim - anchorsim, min=0.0)
    loss = loss + I2A_simdif
    loss = loss.mean() / n
    return loss


class MultiModalMMNet(pl.LightningModule):
    def __init__(self, hparams):
        super(MultiModalMMNet, self).__init__()
        if not isinstance(hparams, Namespace):
            hparams = dotdict(hparams)
        self.save_hyperparameters(hparams)

        self.num_classes = self.hparams.num_classes

        self.videonet = ResNet3dVideo.load_from_checkpoint(
            self.hparams.video_pretrained_folder
            + "/VideoNet-fold"
            + str(self.hparams.fold)
            + ".ckpt",
            video_emb=1,
            get_mm=1,
            strict=False,
        )
        self.audio_dave = DaveAudionet.load_from_checkpoint(
            self.hparams.audio_pretrained_folder
            + "/AudioNet-fold"
            + str(self.hparams.fold)
            + ".ckpt",
            audio_emb=1,
        )
        self.fc = nn.Linear(490, self.num_classes)

    def forward(self, x):
        x_audio = x[0]
        x_video = x[1]
        x_audio = self.audio_dave(x_audio)
        x_audio = x_audio.squeeze(2).squeeze(2)
        # video
        x_video = self.videonet(x_video)
        x_video = x_video.squeeze(2)
        x_res = batch_computeMatchmap(x_video, x_audio)
        if self.hparams.multimodal_emb == 0:
            x_res = x_res.view(x_res.shape[0], -1)
            x_res = self.fc(x_res)
        return x_res, (x_video, x_audio)

    def training_step(self, batch, batch_idx):
        x_audio, x_video, y = batch["audio"], batch["video"], batch["label"]
        y_hat, X_v_a = self.forward((x_audio, x_video))
        return y_hat, y, X_v_a

    def training_step_end(self, batch_parts_outputs):
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
            X_v_a = torch.cat(batch_parts_outputs[2], dim=1)

        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]
            X_v_a = batch_parts_outputs[2]
        if self.hparams.use_matchmap_loss:
            loss = F.cross_entropy(y_hat, y) + batch_sampled_margin_rank_loss(
                X_v_a[0], X_v_a[1], simtype=self.hparams.simtype, labels=y
            ) * (self.hparams.task_weight)
        else:
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
        y_hat, X_v_a = self.forward((x_audio, x_video))
        return y_hat, y, X_v_a

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
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x_audio, x_video, y = batch["audio"], batch["video"], batch["label"]
        y_hat, X_v_a = self.forward((x_audio, x_video))
        return y_hat, y, X_v_a

    def test_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch
        if batch_parts_outputs[0].ndim > 3:
            y_hat = torch.cat(batch_parts_outputs[0], dim=1)
            y = torch.cat(batch_parts_outputs[1], dim=1)
            X_M = torch.cat(batch_parts_outputs[2], dim=1)

        else:
            y_hat = batch_parts_outputs[0]
            y = batch_parts_outputs[1]
            X_M = batch_parts_outputs[2]

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
        #

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
        # REQUIRED
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

        parser.add_argument("--use_matchmap_loss", default=1, type=int)
        parser.add_argument("--simtype", default="SIMA", type=str)
        parser.add_argument("--aux_task_weight", default=0.078, type=float)
        parser.add_argument("--multimodal_emb", default=0, type=int)

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=10000, type=int)
        parser.add_argument("--num_classes", dest="num_classes", default=4, type=int)

        return parser
