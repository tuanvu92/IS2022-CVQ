import torch
from torch.utils.data import Dataset
from utils.common_utils import get_list_of_files
import numpy as np
import random
import soundfile as sf


class AudioNoiseDataset(Dataset):
    def __init__(self, clean_dir_path, load_noisy=False):
        super().__init__()
        self.load_noisy = load_noisy
        self.file_list = get_list_of_files(clean_dir_path)

    @staticmethod
    def scale_db(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        clean_wav_fp = self.file_list[index]
        audio_clean, _ = sf.read(clean_wav_fp)
        audio_clean, _, scale = self.scale_db(audio_clean,
                                              target_dB_FS=np.random.uniform(-35, -20, 1))

        if self.load_noisy:
            noisy_wav_fp = clean_wav_fp.replace("clean", "noisy")
            audio_noise, _ = sf.read(noisy_wav_fp)
            audio_noise *= scale
            return audio_clean, audio_noise
        else:
            return audio_clean


class AudioNoiseCollateFn(object):
    def __init__(self, max_batch_len=256, load_noisy=False):
        self.max_batch_len = max_batch_len * 100 - 300
        self.load_noisy = load_noisy

    def __call__(self, batch):
        batch_size = len(batch)
        if self.load_noisy:
            clean_batch = np.zeros((batch_size, self.max_batch_len))
            noisy_batch = np.zeros((batch_size, self.max_batch_len))
            for i, (x_clean, x_noise) in enumerate(batch):
                if len(x_clean) > self.max_batch_len:
                    start = random.randint(0, len(x_clean) - self.max_batch_len)
                    clean_batch[i] = x_clean[start: start + self.max_batch_len]
                    noisy_batch[i] = x_noise[start: start + self.max_batch_len]
                else:
                    clean_batch[i] = np.pad(x_clean, [0, self.max_batch_len-len(x_clean)], "wrap")
                    noisy_batch[i] = np.pad(x_noise, [0, self.max_batch_len-len(x_clean)], "wrap")
            return torch.from_numpy(clean_batch).float(), torch.from_numpy(noisy_batch).float()
        else:
            clean_batch = np.zeros((batch_size, self.max_batch_len))
            for i, x_clean in enumerate(batch):
                if len(x_clean) > self.max_batch_len:
                    start = random.randint(0, len(x_clean) - self.max_batch_len)
                    clean_batch[i] = x_clean[start: start + self.max_batch_len]
                else:
                    clean_batch[i] = np.pad(x_clean, [0, self.max_batch_len - len(x_clean)], "wrap")
            return torch.from_numpy(clean_batch).float()
