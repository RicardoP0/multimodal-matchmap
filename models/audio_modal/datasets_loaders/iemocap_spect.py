#%%
from torch.utils.data import Dataset
import torch
import pandas as pd
import joblib

def permute_audio(x):
    return x.permute(1, 2, 0)
class IEMOCAPSpectDataset(Dataset):
    def __init__(
        self, root_dir, set_type="train", transform=None, num_classes=6, fold=1, data_df=None
    ):
        """
        Args:
            audio_dir (string): Directory with all the images.
            set_type (string): Data partition to use. Possible values are train, val, test.
            transform (torchvision.transforms) 
            fold  (int): Fold to use in 10fold CV
            data_df (pandas Dataframe): Dataframe to use in dataloader. If None, the csv from audio_dir will be used.
        """
        self.root_dir = root_dir + "/"
        self.transform = transform
        self.set_type = set_type
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
            print(self.data_df.shape[0], self.data_df.emotion.value_counts())
        self.length = self.data_df.shape[0]
        print(self.length, set_type)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        wav_f = self.data_df.loc[idx].wav_file
        label = self.emotion_dict[self.data_df.loc[idx].emotion]
        folder = "/" + wav_f[:-5] + "/"
        f = self.data_df.loc[idx].audio_spect
        x = joblib.load(self.root_dir + "/" + folder + f)
        if x.ndim == 4:
            X_res = []
            for i in range(x.shape[0]):
                X_res.append(self.transform(x[i, :, :, :]))
            X_res = torch.stack(X_res, dim=0)
            X_res = X_res.permute(2, 0, 3, 1)
        else:
            X_res = self.transform(x)
        if X_res.dtype == torch.float64:
            X_res = X_res.type(torch.float32)
        return X_res, label
