#%%
from torch.utils.data import Dataset
import torchvision
import torch
import pandas as pd
import torchvision.transforms as transforms


class IEMOCAPFaceDataset(Dataset):
    def __init__(
        self,
        root_dir,
        set_type="train",
        transform=None,
        temporal_transform=None,
        num_classes=4,
        fold=1,
        data_df=None,
    ):
        """
        Args:
            video_dir (string): Directory with all the images.
            set_type (string): Data partition to use. Possible values are train, val, test.
            transform (torchvision.transforms): Transformations from _transforms_video or others that accept a video tensor .
            temporal_transform: Function to pick frames from the video.
            fold  (int): Fold to use in 10fold CV
            data_df (pandas Dataframe): Dataframe to use in dataloader. If None, the csv from audio_dir will be used.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.fixed_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()]
        )

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
            self.data_df = pd.read_csv(self.root_dir + "/df_iemocap_final.csv")

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
            self.data_df = self.data_df.drop(self.data_df.columns[:2], axis=1)
        self.length = self.data_df.shape[0]
        self.folder_dict = {
            "Ses01": "Session1",
            "Ses02": "Session2",
            "Ses03": "Session3",
            "Ses04": "Session4",
            "Ses05": "Session5",
        }
        print(self.length, set_type)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data_df.loc[idx].wav_file
        file_idx = self.data_df.loc[idx].audio_spect.split("_")[-2]
        label = self.emotion_dict[self.data_df.loc[idx].emotion]

        folder = (
            self.folder_dict[filename[:5]] + "/" + filename + "_" + file_idx + ".avi"
        )

        start = 0.0
        file_part = int(self.data_df.loc[idx].audio_spect.split("_")[-1][0])
        # Audio files are separated in 3 sec chunks with names corresponding to the segment they belong.
        if file_part != 0:
            # Set start depending on the corresponding audio segment
            start = 3.0 * file_part
        frames, _, _ = torchvision.io.read_video(
            self.root_dir + "/" + folder, start_pts=start, pts_unit="sec"
        )
        if self.temporal_transform:
            frames = self.temporal_transform(frames)
        if self.transform:
            X = []
            frames = self.transform(frames)
        else:
            frames = frames.permute(0, 3, 1, 2)
            temp = []
            for i in range(frames.shape[0]):
                temp.append(self.fixed_transform(frames[i]))
            frames = torch.stack(temp, dim=0)
        return frames, label
