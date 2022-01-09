import os
import torch
from torch.utils.data import Dataset
from utils.common_utils import get_list_of_files, read_file_list
import librosa
import numpy as np
from pydub import AudioSegment
import json
import random
from scipy.io import wavfile
import glob
from scipy import signal
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, wav_dir, n_speaker=None, speaker_id_pos=-2):
        wav_file_list = get_list_of_files(wav_dir)
        self.wav_file_list = [s for s in wav_file_list if self.file_filters(s)]
        self.speaker_label = self.create_speaker_label(n_speaker=n_speaker,
                                                       speaker_id_pos=speaker_id_pos)
        # self.wav_file_list = [fname for fname in self.wav_file_list
        #                       if fname.split("/")[speaker_id_pos] in self.speaker_label]
        self.wav_file_list.sort()
        self.wav_file_list.reverse()
        self.n_speaker = n_speaker

    @staticmethod
    def file_filters(fname):
        ret = True
        if fname.find(".wav") == -1 and fname.find(".m4a") == -1:
            ret = False
        return ret

    def get_audio(self, idx):
        # x, fs = librosa.load(self.wav_file_list[idx], sr=None)
        audio = AudioSegment.from_file(self.wav_file_list[idx])
        x = np.array(audio.get_array_of_samples()).astype(np.float32)
        fs = audio.frame_rate

        x = x[:20*fs]
        assert fs == 16000

        try:
            x = x / max(abs(x))
        except ValueError:
            print(self.wav_file_list[idx])
            print(len(x))
            print(abs(x))

        return torch.from_numpy(x).type(torch.float)

    def get_label(self, fname):
        speaker_name = fname.split("/")[-1].split("_")[0]
        return self.speaker_label[speaker_name]

    def create_speaker_label(self, n_speaker=None, speaker_id_pos=-2):
        speaker_list = []
        for fname in self.wav_file_list:
            speaker_name = fname.split("/")[speaker_id_pos]
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
        speaker_list.sort()
        if n_speaker is not None:
            speaker_list = speaker_list[:n_speaker]
        speaker_label = {spkr_name: i for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_label

    def __getitem__(self, index):
        if self.n_speaker is not None:
            y = torch.zeros([1, self.n_speaker])
            speaker_label = self.get_label(self.wav_file_list[index])
            y[0, speaker_label] = 1.0
            return self.get_audio(index), self.wav_file_list[index], y
        else:
            return self.get_audio(index), self.wav_file_list[index], torch.tensor([0, 0])

    def __len__(self):
        return len(self.wav_file_list)

    @staticmethod
    def mel2mcc(mel):
        c = np.fft.irfft(mel, axis=1)
        c[:, 0, :] /= 2.0
        return c[:, :mel.shape[1], :]

    @staticmethod
    def mcc2mel(mcc):
        sym_c = np.zeros([mcc.shape[0], 2*(mcc.shape[1]-1), mcc.shape[2]])
        sym_c[:, 0, :] *= 2.0
        for i in range(1, mcc.shape[1]):
            sym_c[:, i, :] = mcc[:, i, :]
            sym_c[:, -i, :] = mcc[:, i, :]

        mel = np.fft.rfft(sym_c, axis=1).real
        return mel


class AudioCollateFn(object):
    def __init__(self, onehot=False):
        self.onehot = onehot
        return

    def __call__(self, batch):
        batch_size = len(batch)
        audio_path = [_item[1] for _item in batch]
        audio_len = [x[0].shape[0] for x in batch]

        y = torch.cat([x[2] for x in batch], dim=0)

        max_audio_len = max(audio_len)
        # if max_audio_len < 256*512:
        #     max_audio_len = 256*512
        # max_audio_len = 8*256*(max_audio_len//(8*256) + 1)
        padded_audio = torch.zeros([batch_size, max_audio_len])

        for i, x in enumerate(batch):
            padded_audio[i, :x[0].shape[0]] = x[0]
            audio_len[i] = x[0].shape[0]
        if self.onehot:
            return padded_audio, audio_len, audio_path, y
        else:
            return padded_audio, audio_len, audio_path


class AudioAugmentDataset(Dataset):
    def __init__(self, file_list: str, musan_path: str, rir_path: str,
                 max_frames=200, speaker_label=None, augmentation=True):
        self.file_list = read_file_list(file_list)
        if speaker_label is None:
            print("Creating speaker label...")
            self.speaker_list, self.speaker_label, self.speaker_file_list = self.create_speaker_label()
        else:
            self.speaker_label = speaker_label
            self.speaker_list = list(speaker_label.keys())
            for k in speaker_label:
                self.speaker_list[speaker_label[k]] = k
            self.speaker_file_list = dict()
            random.shuffle(self.file_list)
            for fname in self.file_list:
                speaker_name = self.get_speaker_name(fname)
                if speaker_name not in self.speaker_file_list:
                    self.speaker_file_list[speaker_name] = [fname]
                else:
                    self.speaker_file_list[speaker_name].append(fname)

        self.augmentation = augmentation
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240
        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
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

    def __getitem__(self, index):
        wav_name = self.file_list[index]
        label = self.get_label(wav_name)
        audio = self.load_wav(wav_name, self.max_frames, eval_mode=False)
        if self.augmentation:
            aug_type = random.randint(0, 4)
            if aug_type == 1:
                audio = self.reverberate(audio)
            elif aug_type == 2:
                audio = self.additive_noise('music', audio)
            elif aug_type == 3:
                audio = self.additive_noise('speech', audio)
            elif aug_type == 4:
                audio = self.additive_noise('noise', audio)
        return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.long)

    def create_speaker_label(self):
        speaker_list = []
        speaker_file_list = dict()
        for fname in tqdm(self.file_list):
            speaker_name = self.get_speaker_name(fname)
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
            if speaker_name not in speaker_file_list:
                speaker_file_list[speaker_name] = [fname]
            else:
                speaker_file_list[speaker_name].append(fname)
        speaker_list.sort()
        speaker_label = {spkr_name: int(i) for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_list, speaker_label, speaker_file_list

    @staticmethod
    def get_speaker_name(fname):
        if fname.find("zalo") != -1:
            return fname.split("/")[-2]
        if fname.find("vox") != -1:
            return fname.split("/")[-3]

    def get_label(self, fname):
        speaker_name = self.get_speaker_name(fname)
        return self.speaker_label[speaker_name]

    def additive_noise(self, noise_type: str, audio):
        """ @:param noise_type: one of ["noise", "speech", "music"] """
        assert noise_type in self.noise_types
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        noise_list = self.noise_list[noise_type]
        num_noise = self.num_noise[noise_type]
        noise_list = random.sample(noise_list, random.randint(num_noise[0], num_noise[1]))

        noises = []

        for noise in noise_list:
            noise_audio = self.load_wav(noise, self.max_frames, eval_mode=False)
            noise_snr = random.uniform(self.noise_snr[noise_type][0], self.noise_snr[noise_type][1])
            noise_db = 10 * np.log10(np.mean(noise_audio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise_audio)

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        fs, rir = wavfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]

    @staticmethod
    def load_wav(filename, max_frames, eval_mode=True, num_eval=10):
        # Maximum audio length
        max_audio = max_frames * 160 + 240

        # Read wav file and convert to torch tensor
        sample_rate, audio = wavfile.read(filename)

        audio_size = audio.shape[0]

        if audio_size <= max_audio:
            shortage = max_audio - audio_size + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audio_size = audio.shape[0]

        if eval_mode:
            start_frame = np.linspace(0, audio_size - max_audio, num=num_eval)
        else:
            start_frame = np.array([np.int64(random.random() * (audio_size - max_audio))])

        feats = []
        if eval_mode and max_frames == 0:
            feats.append(audio)
        else:
            for asf in start_frame:
                feats.append(audio[int(asf):int(asf) + max_audio])
        feat = np.stack(feats, axis=0).astype(np.float)
        return feat
