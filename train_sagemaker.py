import os
import sys
import subprocess
import sys


def install():
    # update sagemaker instances' torch version
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "horovod", "-y"])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==1.8.1",
            "torchvision",
            "torchaudio",
        ]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch_lightning"])


# install()

import logging
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger

from models.audio_modal.davenet_audio import DaveAudionet
from models.video_modal.resnet3d import ResNet3dVideo

from models.multimodal.base.multimodalnet_concat import MultiModalConcatNet
from models.multimodal.base.multimodalnet_matchmap import MultiModalMMNet

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, plugins

seed_everything(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)


def main(hparams, network):
    # init module

    model = network(hparams)
    project_folder = "emotion_iemocap"
    checkpoint_path = "/opt/ml/checkpoints/"
    if os.path.isfile(checkpoint_path + "/wandb_id.txt"):
        with open(checkpoint_path + "/wandb_id.txt", "r") as text_file:
            id_wandb = text_file.readline()

    if os.path.isfile(checkpoint_path + "/last.ckpt"):
        print("RESUMING CHECKPOINT")
        resume = checkpoint_path + "/last.ckpt"

    else:
        resume = None
        if os.getenv("NODE_RANK", 0) == 0 and os.getenv("LOCAL_RANK", 0) == 0:
            print("CREATING CHECKPOINT")
            id_wandb = wandb.util.generate_id()
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            with open(checkpoint_path + "/wandb_id.txt", "w") as text_file:
                text_file.write(id_wandb)
    wandb_logger = WandbLogger(
        name=hparams.model_name,
        project=project_folder,
        entity="thesis",
        offline=False,
        id=id_wandb,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.early_stop_num,
        verbose=False,
        mode="min",
    )
    backend = "ddp"  # None#'dp'

    trainer = Trainer(
        max_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        num_nodes=hparams.nodes,
        logger=wandb_logger,
        # weights_summary='full',
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
        # profiler=True,
        benchmark=bool(hparams.fixed_data),
        log_gpu_memory="all",
        deterministic=False,
        precision=hparams.precision,
        default_root_dir=hparams.output_data_dir,
        checkpoint_callback=True,
        resume_from_checkpoint=resume,
        distributed_backend=backend,
        auto_select_gpus=True,
        accumulate_grad_batches=hparams.accum_grad_batches,
        plugins=[plugins.DDPPlugin(find_unused_parameters=True)],
        gradient_clip_val=hparams.gradient_clip_val,
    )

    trainer.fit(model)
    # load best model
    if trainer.is_global_zero and hparams.gpus == 1:
        trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--early_stop_num", type=int, default=10)
    parser.add_argument("--fixed-data", type=int, default=1)
    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument("--sweep-name", type=str, default="")

    # sagemaker params
    parser.add_argument(
        "-o", "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )

    parser.add_argument(
        "-aroot",
        "--audio-folder",
        type=str,
        default=os.environ["SM_CHANNEL_AUDIO_FOLDER"],
    )
    parser.add_argument(
        "-vroot",
        "--video-folder",
        type=str,
        default=os.environ["SM_CHANNEL_VIDEO_FOLDER"],
    )

    if "SM_CHANNEL_AUDIO_FOLDER_2" in os.environ:
        parser.add_argument(
            "-aroot2",
            "--audio-folder-2",
            type=str,
            default=os.environ["SM_CHANNEL_AUDIO_FOLDER_2"],
        )
        parser.add_argument(
            "-vroot2",
            "--video-folder-2",
            type=str,
            default=os.environ["SM_CHANNEL_VIDEO_FOLDER_2"],
        )
    if "SM_CHANNEL_AUDIO_PRETRAINED_FOLDER" in os.environ:
        parser.add_argument(
            "--audio_pretrained_folder",
            type=str,
            default=os.environ["SM_CHANNEL_AUDIO_PRETRAINED_FOLDER"],
        )
        parser.add_argument(
            "--video_pretrained_folder",
            type=str,
            default=os.environ["SM_CHANNEL_VIDEO_PRETRAINED_FOLDER"],
        )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=os.environ["SM_HP_GRADIENT_CLIP_VAL"]
    )

    parser.add_argument("--fold", type=int, default=os.environ["SM_HP_FOLD"])
    parser.add_argument("--dataset", type=str, default=os.environ["SM_HP_DATASET"])
    parser.add_argument(
        "--model-type", type=str, default=os.environ["SM_HP_MODEL_TYPE"]
    )
    parser.add_argument(
        "--model-load-from-checkpoint",
        type=int,
        default=os.environ["SM_HP_MODEL_LOAD_FROM_CHECKPOINT"],
    )

    model_name = os.environ["SM_HP_MODEL_TYPE"]
    if "SM_CHANNEL_MULTIMODAL_PRETRAINED_BASE" in os.environ:
        parser.add_argument(
            "--multimodal_pretrained_base",
            type=str,
            default=os.environ["SM_CHANNEL_MULTIMODAL_PRETRAINED_BASE"],
        )
    if "SM_CHANNEL_MULTIMODAL_PRETRAINED_FOLDER" in os.environ:
        parser.add_argument(
            "--multimodal_pretrained_folder",
            type=str,
            default=os.environ["SM_CHANNEL_MULTIMODAL_PRETRAINED_FOLDER"],
        )

    if model_name == "concat":
        network = MultiModalConcatNet
    elif model_name == "matchmap":
        network = MultiModalMMNet
    elif model_name == "video":
        network = ResNet3dVideo
    elif model_name == "audio":
        network = DaveAudionet

    parser = network.add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    print(hparams)
    main(hparams, network)

