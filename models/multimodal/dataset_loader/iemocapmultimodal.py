#%%

from torch.utils.data import Dataset
import torch
import pandas as pd

from models.video_modal.datasets_loaders.iemocap_video import IEMOCAPFaceDataset
from models.audio_modal.datasets_loaders.iemocap_spect import IEMOCAPSpectDataset


class IEMOCAPMultiModalDataset(Dataset):
    def __init__(
        self,
        audio_dir,
        video_dir,
        set_type="train",
        audio_transform=None,
        video_transform=None,
        vtemporal_transform=None,
        num_classes=4,
        fold=1,
        data_df=None,
    ):
        """
        Args:
            audio_dir (string): Directory with all the images.
            set_type (string): Data partition to use. Possible values are train, val, test.
            audio_transform (torchvision.transforms) 
            video_transform (torchvision.transforms): Transformations from _transforms_video or others that accept a video tensor .
            vtemporal_transform: Function to pick frames from the video.
            fold  (int): Fold to use in 10fold CV
            data_df (pandas Dataframe): Dataframe to use in dataloader. If None, the csv from audio_dir will be used.
        """
        self.audio_dir = audio_dir + "/"
        self.video_dir = video_dir + "/"

        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.vtemporal_transform = vtemporal_transform
        self.num_classes = num_classes
        if num_classes == 8:
            self.emotion_dict = {
                "ang": 0,
                "hap": 1,
                "exc": 2,
                "sad": 3,
                "fru": 4,
                "fea": 5,
                "sur": 6,
                "neu": 7,
            }
        elif num_classes == 6:
            self.emotion_dict = {
                "ang": 0,
                "hap": 1,
                "exc": 2,
                "sad": 3,
                "fru": 4,
                "neu": 5,
            }
        elif num_classes == 4:
            self.emotion_dict = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}
        if data_df is not None:
            self.data_df = data_df
        else:
            self.data_df = pd.read_csv(self.audio_dir + "/df_iemocap_final.csv")
            fold_dict = {
                1: [
                    ["session_1", "session_2", "session_3", "session_4"],
                    ["session_5"],
                    ["_F", "_M"],
                ],
                2: [
                    ["session_1", "session_2", "session_3", "session_5"],
                    ["session_4"],
                    ["_F", "_M"],
                ],
                3: [
                    ["session_1", "session_2", "session_4", "session_5"],
                    ["session_3"],
                    ["_F", "_M"],
                ],
                4: [
                    ["session_1", "session_4", "session_3", "session_5"],
                    ["session_2"],
                    ["_F", "_M"],
                ],
                5: [
                    ["session_4", "session_2", "session_3", "session_5"],
                    ["session_1"],
                    ["_F", "_M"],
                ],
                6: [
                    ["session_1", "session_2", "session_3", "session_4"],
                    ["session_5"],
                    ["_M", "_F"],
                ],
                7: [
                    ["session_1", "session_2", "session_3", "session_5"],
                    ["session_4"],
                    ["_M", "_F"],
                ],
                8: [
                    ["session_1", "session_2", "session_4", "session_5"],
                    ["session_3"],
                    ["_M", "_F"],
                ],
                9: [
                    ["session_1", "session_4", "session_3", "session_5"],
                    ["session_2"],
                    ["_M", "_F"],
                ],
                10: [
                    ["session_4", "session_2", "session_3", "session_5"],
                    ["session_1"],
                    ["_M", "_F"],
                ],
            }
            if set_type == "train":
                self.data_df = self.data_df[
                    self.data_df["session"].isin(fold_dict[fold][0])
                ]
            elif set_type == "val":
                self.data_df = self.data_df[
                    self.data_df["session"].isin(fold_dict[fold][1])
                ]
                self.data_df = self.data_df[
                    self.data_df["wav_file"].str.contains(fold_dict[fold][2][0])
                ]

            else:
                self.data_df = self.data_df[
                    self.data_df["session"].isin(fold_dict[fold][1])
                ]
                self.data_df = self.data_df[
                    self.data_df["wav_file"].str.contains(fold_dict[fold][2][1])
                ]

            if num_classes == 4:
                self.data_df.loc[self.data_df.emotion == "exc", "emotion"] = "hap"

            self.data_df = self.data_df[
                self.data_df["emotion"].isin(self.emotion_dict.keys())
            ]
            self.data_df = self.data_df.reset_index()
        self.length = self.data_df.shape[0]
        self.folder_dict = {
            "Ses01": "Session1",
            "Ses02": "Session2",
            "Ses03": "Session3",
            "Ses04": "Session4",
            "Ses05": "Session5",
        }
        print(self.length, set_type)

        self.audio_dataset = IEMOCAPSpectDataset(
            self.audio_dir,
            set_type=set_type,
            transform=self.audio_transform,
            num_classes=num_classes,
            fold=fold,
            data_df=self.data_df,
        )
        self.video_dataset = IEMOCAPFaceDataset(
            self.video_dir,
            set_type=set_type,
            transform=video_transform,
            temporal_transform=vtemporal_transform,
            num_classes=num_classes,
            fold=fold,
            data_df=self.data_df,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.emotion_dict[self.data_df.loc[idx].emotion]
        X_audio, l1 = self.audio_dataset[idx]
        X_video, l2 = self.video_dataset[idx]

        return {"audio": X_audio, "video": X_video, "label": label}
