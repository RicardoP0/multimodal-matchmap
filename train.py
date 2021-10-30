"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.audio_modal.davenet_audio import DaveAudionet
from models.video_modal.resnet3d import ResNet3dVideo

from models.multimodal.base.multimodalnet_concat import MultiModalConcatNet
from models.multimodal.base.multimodalnet_matchmap import MultiModalMMNet

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, plugins

seed_everything(42)


def main(hparams, network):
    # init module

    model = network(hparams)

    project_folder = "emotion_iemocap"
    checkpoint_path = "checkpoints/" + hparams.model_name
    wandb_logger = WandbLogger(
        name=hparams.model_name, project=project_folder, offline=False
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
    )

    trainer = Trainer(
        max_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        num_nodes=hparams.nodes,
        logger=wandb_logger,
        callbacks=[
            early_stop_callback,
            ModelCheckpoint(
                dirpath=checkpoint_path + "/",
                filename=hparams.model_name + "_{epoch:02d}-{val_loss:.2f}",
                save_last=True,
                save_top_k=4,
                monitor="val_loss",
                mode="min",
                every_n_val_epochs=1,
            ),
        ],
        benchmark=bool(hparams.fixed_data),
        log_gpu_memory="all",
        precision=hparams.precision,
        default_root_dir=hparams.output_data_dir,
        checkpoint_callback=True,
        distributed_backend="ddp",
        accumulate_grad_batches=hparams.accum_grad_batches,
        gradient_clip_val=3.0,
        plugins=[plugins.DDPPlugin(find_unused_parameters=True)],
    )

    if os.getenv("NODE_RANK", 0) == 0 and os.getenv("LOCAL_RANK", 0) == 0:
        wandb_logger.experiment.config.update({"dataset": "IEMOCAP_SPECT_FRAMES"})

    trainer.fit(model)
    if hasattr(trainer.checkpoint_callback, "best_model_path"):
        print(trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    fold = "1"

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="TEST" + fold)  # )
    parser.add_argument("--early_stop_num", type=int, default=3)
    parser.add_argument("--fixed-data", type=int, default=1)
    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument("--sweep-name", type=str, default="sweep")
    parser.add_argument("-o", "--output-data-dir", type=str, default="../models")

    # training data
    parser.add_argument("-aroot", "--audio-folder", type=str)
    parser.add_argument("-vroot", "--video-folder", type=str)

    parser.add_argument("--audio_pretrained_folder", type=str)
    parser.add_argument("--video_pretrained_folder", type=str)
    parser.add_argument("--multimodal_pretrained_folder", type=str)
    parser.add_argument("--multimodal_pretrained_base", type=str)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model-type", type=str)
    parser.add_argument("--model-load-from-checkpoint", type=int)

    network = MultiModalMMNet
    parser = network.add_model_specific_args(parser)

    root = ""
    root = ""

    dataset = "iemocap"

    model_type = "matchmap"
    model_load_from_checkpoint = "0"
    num_classes = "4"

    if dataset == "iemocap" or dataset == "iemocap_seq":
        audio_folder = "/datasets/IEMOCAP/LOGMEL_DELTAS"
        video_folder = "/datasets/IEMOCAP/TRAINING_VIDEOS_2"

    hparams = parser.parse_args(
        [
            "--audio-folder",
            root + audio_folder,
            "--video-folder",
            root + video_folder,
            "--audio_pretrained_folder",
            root + "/datasets/IEMOCAP/MODELS/AudioModal",
            "--dataset",
            dataset,
            "--video_pretrained_folder",
            root + "/datasets/IEMOCAP/MODELS/VideoModal",
            "--max_nb_epochs",
            "100",
            "--fold",
            fold,
            "--num_classes",
            num_classes,
            "--batch_size",
            "32",
            "--model-type",
            model_type,
            "--model-load-from-checkpoint",
            model_load_from_checkpoint,
        ]
    )
    main(hparams, network)

