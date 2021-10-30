#%%
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/kenshohara/3D-ResNets-PyTorch
"""
torch.Size([1, 3, 16, 112, 112])
torch.Size([1, 64, 16, 56, 56])
torch.Size([1, 64, 8, 28, 28])
torch.Size([1, 64, 8, 28, 28])
torch.Size([1, 128, 4, 14, 14])
torch.Size([1, 256, 2, 7, 7])
torch.Size([1, 512, 1, 4, 4])
torch.Size([1, 512, 1, 1, 1])
"""
#%%
def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        zero_pads = zero_pads.to(out.device)
        # print(out.dtype, zero_pads.dtype)
        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


# %%
import sklearn.metrics as metrics
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as video_transforms

import torchvision as torchvision
from argparse import ArgumentParser, Namespace

#%%
from utils.dotdict import dotdict
from models.video_modal.datasets_loaders.iemocap_video import IEMOCAPFaceDataset
from models.video_modal.data_aug.temporal_transforms import (
    TemporalCenterCrop,
    TemporalRandomCrop,
    TemporalEvenSample,
)


import pytorch_lightning.metrics.functional as F_m
import pandas as pd
import wandb
from torchmetrics.functional.classification.f_beta import f1
import torchmetrics


class ResNet3dVideo(pl.LightningModule):
    def __init__(self, hparams):
        super(ResNet3dVideo, self).__init__()
        if not isinstance(hparams, Namespace):
            hparams = dotdict(hparams)
        self.save_hyperparameters(hparams)
        self.num_classes = self.hparams.num_classes
        self.example_input_array = torch.zeros(1, 3, 16, 112, 112)
        models_files = {
            18: ("/r3d18_KM_200ep.pth", 1039, 8192),
            34: ("/r3d34_KM_200ep.pth", 1039, 8192),
            50: ("/r3d50_KMS_200ep.pth", 1139, 32768),
            101: ("/r3d101_KM_200ep.pth", 1039, 32768),
        }
        if self.hparams.resnet_type == 0:
            self.net = torchvision.models.video.r3d_18(
                pretrained=self.hparams.pretrained_video_resnet
            )  # generate_model(self.hparams.model_depth, n_classes=models_files[self.hparams.model_depth][1])
        elif self.hparams.resnet_type == 1:
            self.net = torchvision.models.video.mc3_18(
                pretrained=self.hparams.pretrained_video_resnet
            )  # generate_model(self.hparams.model_depth, n_classes=models_files[self.hparams.model_depth][1])
        else:
            self.net = torchvision.models.video.r2plus1d_18(
                pretrained=self.hparams.pretrained_video_resnet
            )  # generate_model(self.hparams.model_depth, n_classes=models_files[self.hparams.model_depth][1])

        print("video hparams")
        print(self.hparams)
        if self.hparams.video_emb == 1:
            if self.hparams.get_mm == 1:
                tmp = list(self.net.children())[:-2]
                tmp.append(nn.AdaptiveAvgPool3d((1, 7, 7)))
                self.net = nn.Sequential(*tmp)
            else:
                self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.fc = nn.Identity()
        else:
            tmp = list(self.net.children())[:-1]
            tmp.append(nn.Dropout(self.hparams.dropout))
            self.net = nn.Sequential(*tmp)
            self.fc = nn.Linear(512, self.hparams.num_classes)

    def forward(self, x):
        x = self.net(x)
        if self.hparams.video_emb == 0:
            x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
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
        print(y_hat, y)
        loss_val = F.cross_entropy(y_hat, y)
        y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
        acc_micro = torchmetrics.functional.precision(
            y_pred, y, average="micro", num_classes=self.num_classes
        )
        acc_macro = torchmetrics.functional.precision(
            y_pred, y, average="macro", num_classes=self.num_classes
        )
        acc_weighted = torchmetrics.functional.precision(
            y_pred, y, average="weighted", num_classes=self.num_classes
        )
        f1_val = f1(y_pred, y, num_classes=self.num_classes, average="macro")

        return {
            "val_loss": loss_val,
            "val_f1": f1_val,
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

        for key, val in tqdm_dict.items():
            self.log(key, val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
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

        acc_micro = torchmetrics.functional.precision(
            y_pred, y, average="micro", num_classes=self.num_classes
        )
        acc_macro = torchmetrics.functional.precision(
            y_pred, y, average="macro", num_classes=self.num_classes
        )
        acc_weighted = torchmetrics.functional.precision(
            y_pred, y, average="weighted", num_classes=self.num_classes
        )

        f1_micro = f1(y_pred, y, num_classes=self.num_classes, average="micro")
        f1_macro = f1(y_pred, y, num_classes=self.num_classes, average="macro")
        f1_weighted = f1(y_pred, y, num_classes=self.num_classes, average="weighted")

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
        # REQUIRED
        if self.hparams.augmentation == 1:
            transform = transforms.Compose(
                [
                    video_transforms.ToTensorVideo(),
                    video_transforms.RandomHorizontalFlipVideo(p=0.3),
                ]
            )
        else:
            transform = transforms.Compose([video_transforms.ToTensorVideo(),])

        temptr = transforms.Compose([TemporalCenterCrop(16)])

        return DataLoader(
            IEMOCAPFaceDataset(
                self.hparams.video_folder,
                set_type="train",
                transform=transform,
                temporal_transform=temptr,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        transform = transforms.Compose([video_transforms.ToTensorVideo(),])

        temptr = transforms.Compose([TemporalCenterCrop(16)])
        return DataLoader(
            IEMOCAPFaceDataset(
                self.hparams.video_folder,
                set_type="val",
                transform=transform,
                temporal_transform=temptr,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        transform = transforms.Compose([video_transforms.ToTensorVideo(),])

        temptr = transforms.Compose([TemporalCenterCrop(16)])

        return DataLoader(
            IEMOCAPFaceDataset(
                self.hparams.video_folder,
                set_type="test",
                transform=transform,
                temporal_transform=temptr,
                num_classes=self.num_classes,
                fold=self.hparams.fold,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--preload_percent", default=0.0, type=float)

        # MODEL specific
        parser.add_argument("--resnet_type", default=1, type=int)
        parser.add_argument("--dropout", default=0.4232, type=float)
        parser.add_argument("--pretrained_video_resnet", default=0, type=int)

        parser.add_argument("--video_emb", default=0, type=int)
        parser.add_argument("--get_mm", default=0, type=int)

        parser.add_argument("--augmentation", default=0, type=int)
        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=10000, type=int)
        parser.add_argument("--num_classes", dest="num_classes", default=4, type=int)
        return parser

