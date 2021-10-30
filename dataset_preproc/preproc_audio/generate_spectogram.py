# %%
import pandas as pd
import librosa
import librosa.display
import os
import numpy as np
import joblib


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def gen_melspect(
    file_path,
    output_name,
    sr=None,
    n_fft=2048,
    n_mels=128,
    win_length=None,
    hop_length=512,
    min_dur=8.0,
    output_length=251,
    image=False,
    dataset="iemocap",
    deltas=False,
    start=None,
    end=None,
    means=None,
    stds=None,
):

    y, sr = librosa.load(file_path, sr=sr)
    if means is not None:
        y = (y - means) / stds
    if start is not None:
        y = y[int(start * sr) : int(end * sr)]

    def pad(a, i):
        return a[0:i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))

    def trim_pad_sample(x):
        samples = []
        duration_s = x.shape[0] / float(sr)
        if duration_s < min_dur:
            samples.append(pad(x, int(sr * min_dur)))

        elif duration_s / min_dur > 2 or (duration_s / min_dur) % 1 > 0.65:
            pos = int(min_dur * sr)
            samples = []
            samples.append(x[:pos])
            x = x[pos:]
            dur_s = x.shape[0] / sr
            if dur_s / min_dur > 2 or (dur_s / min_dur) % 1 > 0.65:

                def append_sample(lst):
                    temp = []
                    for item in lst:
                        if len(item) > 1 and type(item) == list:
                            temp.append(item)
                        else:
                            temp.append(item)
                    return temp

                for item in append_sample(trim_pad_sample(x)):
                    samples.append(item)
        else:

            x = x[: int(min_dur * float(sr))]
            samples.append(x)
        return samples

    if dataset == "iemocap":
        samples = trim_pad_sample(y)
    else:
        duration_s = y.shape[0] / float(sr)
        if duration_s > min_dur:
            y = y[: int(min_dur * sr)]
        samples = [y]

    k = 0
    for item in samples:
        y = item
        res = librosa.feature.melspectrogram(
            y,
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            window="hamming",
            fmin=300,
            fmax=8000,
        )
        res = librosa.power_to_db(res, np.max)

        if res.shape[1] > output_length:
            res = res[:, :output_length]
            # print(mfccs.shape)
        elif res.shape[1] < output_length:
            res = np.pad(res, ((0, 0), (0, output_length - res.shape[1])), "constant")

        if deltas:
            logmel_delta = librosa.feature.delta(res)
            deltadelta = librosa.feature.delta(res, order=2)
            if means is not None:
                res = librosa.util.normalize(res)
                logmel_delta = librosa.util.normalize(logmel_delta)
                deltadelta = librosa.util.normalize(deltadelta)

            res = np.stack([res, logmel_delta, deltadelta])

        joblib.dump(res, output_name.format(k))
        k += 1


# %%


if __name__ == "__main__":
    n_mels = 128  # number of bins in spectrogram. Height of image
    # time_steps = 384  # number of time-steps. Width of image
    n_fft = 2048
    hop_length = 512  # 1524  # number of samples per time-step in spectrogram
    win_length = 128  # n_fft512
    min_dur = 8.0
    dataset = "iemocap"
    grayscale = True
    mlst = []
    if dataset == "iemocap":
        """
        pd.Series(mlst).describe()
        count    2170.000000
        mean        4.379649
        std         3.415235
        min         0.779937
        25%         2.109938
        50%         3.259937
        75%         5.667500
        max        34.138750
        dtype: float64
        """
        # load audio. Using example from librosa
        print(os.getcwd())
        source_path = "IEMOCAP_full_release.tar/IEMOCAP_full_release/Session{}/sentences/wav/"
        dest_path = "datasets/IEMOCAP/LOGMEL_DELTAS/"

        df = pd.read_csv("df_iemocap.csv")
        processed_files = []
        for _, row in df.iterrows():
            if row.name in processed_files:
                continue
            sess_path = source_path.format(row.wav_file[4])
            folder = row.wav_file[:-5]
            source_file = os.path.join(sess_path, folder, row.wav_file + ".wav")

            if not os.path.exists(dest_path + folder):
                os.makedirs(dest_path + folder)

            # print('dest',dest_path + i)
            # print('source',file_path)
            sr = 16000
            preemph_coef = 0.97
            sample_rate = sr
            window_size = 0.025
            window_stride = 0.01
            num_mel_bins = 40

            n_fft = 512  # int(sample_rate * window_size)
            win_length = int(sample_rate * window_size)  # None#
            hop_length = int(sample_rate * window_stride)  # 256#

            same_rows = df[df.wav_file == row.wav_file]
            init_start = 0.0
            for _, i in same_rows.iterrows():
                file_name = i.wav_file + "_" + str(i.name)
                out = dest_path + folder + "/" + file_name + "_{}.joblib"
                end = i.end_time - i.start_time + init_start

                gen_melspect(
                    source_file,
                    out,
                    sr=sr,
                    min_dur=3.0,
                    output_length=300,
                    dataset=dataset,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=num_mel_bins,
                    deltas=True,
                    start=init_start,
                    end=end,
                )
                init_start = end
                processed_files.append(i.name)
