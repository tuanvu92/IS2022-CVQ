import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA
import json
import random
from scipy.io import wavfile


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none', cmap="Blues")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def get_list_of_files(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def read_hdf5(hdf5_name, hdf5_path):
    hdf5_file = h5py.File(hdf5_name, "r")
    if hdf5_path not in hdf5_file:
        print("There is no such a data in hdf5 file. ({hdf5_path})")
        return None
    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()
    return hdf5_data


def read_file_list(fp):
    assert (fp.find(".txt") != -1) or (fp.find(".json") != 1)
    if fp.find(".txt") != -1:
        # TXT format
        with open(fp, "r") as f:
            file_list = f.read().split("\n")
        file_list = [f for f in file_list if f != ""]
    else:
        # JSON format
        with open(fp, "r") as f:
            file_list_dict = json.load(f)
        file_list = [os.path.join(file_list_dict["root_path"], fname) for fname in file_list_dict["file_list"]]
    return file_list


def load_wav(filename, max_frames, eval_mode=True):
    # Maximum audio length
    max_audio = max_frames * 160 + 240
    # Read wav file and convert to torch tensor
    sample_rate, audio = wavfile.read(filename)
    audio = audio / (max(abs(audio))+1)
    audio_size = audio.shape[0]
    if audio_size <= max_audio:
        shortage = max_audio - audio_size + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audio_size = audio.shape[0]

    if eval_mode:
        n_splits = 2 * audio_size // max_audio
        start_frame = np.linspace(0, audio_size - max_audio, num=n_splits)
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