import os
import torch
from torch.utils.data import Dataset
from utils.common_utils import read_file_list
import numpy as np
import random
from scipy.io import wavfile
import glob
from scipy import signal
from tqdm import tqdm
import pyworld
import soundfile as sf


class AudioAugmentDataset(Dataset):
    """ Dataset with augmentation
    """
    def __init__(self, file_list,
                 musan_path: str = "/home/messier/PycharmProjects/data/musan_split/train/",
                 rir_path: str = "/home/messier/PycharmProjects/data/rirs_noises/RIRS_NOISES/simulated_rirs/",
                 max_frames=200,
                 augmentation=True):
        if isinstance(file_list, str):
            self.file_list = read_file_list(file_list)
        elif isinstance(file_list, list):
            self.file_list = file_list
        random.shuffle(self.file_list)
        self.augmentation = augmentation
        self.max_frames = max_frames
        self.max_audio = max_frames * 100 - 300
        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [-5, 20], 'speech': [5, 20], 'music': [0, 20]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noise_list = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if not file.split('/')[-4] in self.noise_list:
                self.noise_list[file.split('/')[-4]] = []
            self.noise_list[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def __len__(self):
        return len(self.file_list)

    def get_data(self, index):
        wav_name = self.file_list[index]
        audio_clean = self.load_wav(wav_name, self.max_frames)
        audio_clean, _, _ = self.scale_db(audio_clean, np.random.uniform(-35, -20, 1))

        if self.augmentation:
            # audio_clean, _, _ = self.scale_db(audio_clean, -25)
            audio_noise = self.augment(audio_clean)
            return torch.from_numpy(audio_clean).float(),\
                torch.from_numpy(audio_noise).float()
        else:
            return torch.from_numpy(audio_clean).float()

    def __getitem__(self, index):
        return self.get_data(index)

    @staticmethod
    def scale_db(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    def additive_noise(self, noise_type: str, audio_clean):
        """ @:param noise_type: one of ["noise", "speech", "music"] """
        assert noise_type in self.noise_types, "Unsupported noise type: %s" % noise_type
        clean_db = 10 * np.log10(np.mean(audio_clean ** 2) + 1e-9)
        noise_list = self.noise_list[noise_type]
        # num_noise = self.num_noise[noise_type]
        noise_fname = random.choice(noise_list)
        noise = self.load_wav(noise_fname, self.max_frames)

        if len(noise) < len(audio_clean):
            noise = np.pad(noise, (0, len(audio_clean) - len(noise)), mode='wrap')
        else:
            noise = noise[:len(audio_clean)]
        # noise, _, _ = self.scale_db(noise)
        noise_snr = random.uniform(self.noise_snr[noise_type][0], self.noise_snr[noise_type][1])
        noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-9)
        noise = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise

        audio_noise = noise + audio_clean
        audio_noise = audio_noise.astype(np.float32)
        # audio_noise, _, _ = self.scale_db(audio_noise, np.random.uniform(-35, -15, 1))
        return audio_noise

    def reverberate(self, audio_clean):
        rir_file = random.choice(self.rir_files)
        fs, rir = wavfile.read(rir_file)
        rir = rir.astype(np.float)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        audio_reverb = signal.convolve(audio_clean, rir, mode='full')[:self.max_audio]
        audio_reverb = audio_reverb.astype(np.float32)
        audio_reverb = audio_reverb / (max(abs(audio_reverb)) + 1)
        return audio_reverb

    def augment(self, audio):
        aug_type = random.randint(2, 4)
        if aug_type == 1:
            audio = self.reverberate(audio)
        elif aug_type == 2:
            audio = self.additive_noise('music', audio)
        elif aug_type == 3:
            audio = self.additive_noise('speech', audio)
        elif aug_type == 4:
            audio = self.additive_noise('noise', audio)
        return audio

    def load_wav(self, filename, max_frames):
        # Maximum audio length
        max_audio = max_frames * 100 - 300
        audio, sample_rate = sf.read(filename)
        # assert sample_rate == 16000
        audio_size = audio.shape[0]
        if audio_size <= max_audio:
            shortage = max_audio - audio_size + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
        start = random.randint(0, len(audio) - max_audio)
        audio = audio[start: start+max_audio]

        return audio

    @staticmethod
    def remove_dc(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        return x - mean

    @staticmethod
    def extract_f0(x, fs=16000):
        x = x.astype(np.float64)
        _f0, t = pyworld.dio(x, fs,
                             f0_floor=75,
                             f0_ceil=400,
                             frame_period=6.25)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        f0[f0 < 1.0] = 1.0
        f0 = np.log2(f0).astype(np.float32)
        return f0


class AudioAugmentCollateFn(object):
    def __init__(self, augmentation=True):
        self.augmentation = augmentation
        return

    def __call__(self, batch):
        if self.augmentation:
            data_clean = [x[0] for x in batch]
            data_noise = [x[1] for x in batch]
            return torch.stack(data_clean, dim=0), torch.stack(data_noise, dim=0)
        else:
            return torch.stack(batch, dim=0)
