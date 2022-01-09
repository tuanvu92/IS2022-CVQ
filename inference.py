import torch
import models
import json
from models import ComplexVQ2
from scipy.io import wavfile
import numpy as np
from utils.common_utils import get_list_of_files
from data_utils.AudioAugmentDataset import AudioAugmentDataset, AudioAugmentCollateFn
import torchaudio
import os
import argparse
from multiprocessing import Pool, freeze_support, RLock
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def noise_reduction_worker(checkpoint_fp, cfg_fp, scp_fp):
    with open(cfg_fp, "r") as f:
        cfg = json.load(f)
    model = ComplexVQ2(**cfg["model_configs"]).eval().cuda()
    checkpoint = torch.load(checkpoint_fp)
    model.copy_state_dict(checkpoint["state_dict"])
    model.eval()
    with open(scp_fp, "r") as f:
        txt = f.read().split("\n")
    for line in txt:
        if line == "":
            continue
        wav_fp = line.split(" ")[-1]
        wav_fn = line.split(" ")[0]
        sample_rate, audio_noise = wavfile.read(wav_fp)
        audio_noise = audio_noise / 32768.0
        audio_noise = audio_noise - np.mean(audio_noise)
        clean_db = 10 * np.log10(np.mean(audio_noise ** 2) + 1e-9)
        if clean_db < -35:
            continue
        audio_noise, _, _ = AudioAugmentDataset.scale_db(audio_noise, -25)
        audio_noise = torch.from_numpy(audio_noise).float().unsqueeze(0).cuda()
        x_enhance, s_enhance, log_var_hat, log_var_noise, snr_pred = model.inference(audio_noise, pretrain=False)
        # s_enhance = s_enhance.squeeze().cpu().detach().numpy()
        x_enhance = x_enhance.detach().squeeze().cpu().numpy()
        x_enhance = (x_enhance * 32768.0).astype(np.int16)
        wavfile.write("data/"+wav_fn+".wav", sample_rate, x_enhance)


def noise_reduction_multi_process(checkpoint_fp, cfg_fp, scp_fp):
    with open(scp_fp, "r") as f:
        file_list = f.read().split("\n")

    argument_list = [file_list[i * l: (i + 1) * l] for i in range(n_process)]
    l = 1 + len(file_list) // n_process
    pool = Pool(processes=n_process,
                initargs=(RLock(),),
                initializer=tqdm.set_lock)

    jobs = [pool.apply_async(convert_flac_worker, args=(i, n, dst_dir))
            for i, n in enumerate(argument_list)]
    pool.close()
    result_list = [job.get() for job in jobs]
    # Important to print these blanks
    print("\n" * (len(argument_list) + 1))

# Example:
# python inference.py --checkpoint=checkpoint_70000.pt --config=cfg/train_config_cvq.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    parser.add_argument('-s', '--scp', type=str, required=True,
                        help='scp file')
    global args
    args = parser.parse_args()

    inference(args.checkpoint, args.config, args.scp)
