# -*- coding: utf-8 -*-
""" Main training script

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0

"""
import sys
import matplotlib
matplotlib.use("Agg")
from os.path import join, exists
import json
import argparse
import subprocess
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from utils.logger import DataLogger
from utils.common_utils import *
from utils.eer import calculate_err
from time import localtime, strftime
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchaudio
from progressbar import *
import models
from data_utils.AudioAugmentDataset import AudioAugmentDataset, AudioAugmentCollateFn
from data_utils.AudioNoiseDataset import AudioNoiseDataset, AudioNoiseCollateFn
from scipy.io import wavfile
import random
from sklearn.decomposition import PCA
from pesq import pesq


def train(model_name, batch_size, train_epoch,
          iters_per_checkpoint, iters_per_eval,
          checkpoint_prefix, pretrain=True, max_batch_len=256,
          augment_start_epoch=100, start_iteration=0, learning_rate=1e-3,
          continue_from_cpt=False, checkpoint_path="",
          decay_rate=0.98, use_fp16=True, seed=12345,
          num_gpus=1, rank=0, group_name=""):
    torch.manual_seed(seed)
    if num_gpus > 1:
        init_distributed(rank=rank, num_gpus=num_gpus, group_name=group_name, **dist_configs)
    timestamp = strftime("%Y%m%d_%H%M_" + checkpoint_prefix, localtime())
    output_path = join("checkpoints/", timestamp)
    train_clean = True
    if pretrain:
        train_clean = False
    print("Pretrain flags: ", pretrain)
    print("Augmentation flags: ", train_clean)

    checkpoint_dict = None
    lr = learning_rate
    if checkpoint_path != "":
        print(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if continue_from_cpt:
            output_path = "/".join(checkpoint_path.split("/")[:-1])
            assert exists(output_path)
            start_iteration = checkpoint_dict["iteration"]

    # dataset = AudioNoiseDataset(clean_dir_path=data_configs["clean_dir_path"], load_noisy=train_clean)
    dataset = AudioAugmentDataset(file_list="file_lists/train_list.txt",
                                  max_frames=256,
                                  augmentation=train_clean)
    if rank == 0:
        print("Checkpoint dir: %s" % output_path)
        if not exists(output_path):
            os.makedirs(output_path)
        subprocess.run(["cp", "-r", args.config, "data_utils", "models", "train_cvq.py", output_path])

    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    print("Data directory: ", data_configs["clean_dir_path"])
    print("No. training data: ", len(dataset))
    print("Augment start epoch: ", augment_start_epoch)
    # collate_fn = AudioNoiseCollateFn(max_batch_len=max_batch_len,
    #                                  load_noisy=train_clean)
    collate_fn = AudioAugmentCollateFn(augmentation=train_clean)

    dataloader = DataLoader(dataset=dataset,
                            sampler=train_sampler,
                            batch_size=batch_size // num_gpus,
                            collate_fn=collate_fn,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            shuffle=True if train_sampler is None else False)
    # ===== Initialize model ======
    model = getattr(models, model_name)(**model_configs)
    if checkpoint_path != "":
        print("Loading model weight from checkpoint ", checkpoint_path)
        model.copy_state_dict(checkpoint_dict["state_dict"])
    model = model.cuda()
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    div_factor=4,
                                                    final_div_factor=10,
                                                    steps_per_epoch=len(dataloader),
                                                    epochs=train_epoch,
                                                    pct_start=0.1)
    start_epoch = 0
    if checkpoint_path != "" and continue_from_cpt:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        scheduler.load_state_dict(checkpoint_dict["scheduler"])
        start_epoch = checkpoint_dict["epoch"]
    print("Start epoch: ", start_epoch)
    print("Learning rate: ", scheduler.get_last_lr())
    logger = None
    validator = None
    if rank == 0:
        logger = DataLogger(logdir=join(output_path, "logs"))
        validator = Validator(logger=logger, start_iteration=start_iteration,
                              pretrain=pretrain, **validation_configs)

    # =====START: ADDED FOR AMP ======
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    # =====END: ADDED FOR AMP ======

    iteration = start_iteration
    for epoch in range(start_epoch, train_epoch):
        try:
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if rank == 0:
                widgets = [FormatLabel(''), Bar('#'), ' ', Percentage(format='%(percentage).1f%%'),
                           " (", Counter(), "|%d) " % len(dataloader), " ", Timer(), " ", ETA()]
                iterator = progressbar(dataloader, redirect_stdout=True, widgets=widgets)
            else:
                iterator = dataloader

            for batch in iterator:
                model.zero_grad()
                optimizer.zero_grad()
                if epoch > augment_start_epoch:
                    augment = True
                else:
                    augment = False

                if use_fp16:
                    with torch.cuda.amp.autocast(enabled=True):
                        if train_clean:
                            x_clean = batch[0].cuda()
                            x_noise = batch[1].cuda()
                            loss_components = model([x_clean, x_noise], train_clean=False)
                        else:
                            loss_components = model([batch.cuda()], augment=augment)
                    loss = loss_components["loss"]
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if train_clean:
                        x_clean = batch[0].cuda()
                        x_noise = batch[1].cuda()
                        loss_components = model([x_clean, x_noise], train_clean=False)
                    else:
                        loss_components = model([batch.cuda()], train_clean=True, augment=augment)
                    loss = loss_components["loss"]
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                if num_gpus > 1:
                    for i in loss_components.keys():
                        loss_components[i] = reduce_tensor(loss_components[i].data, num_gpus).item()
                else:
                    for i in loss_components.keys():
                        loss_components[i] = loss_components[i].item()
                if rank == 0:
                    widgets[0] = FormatLabel("{:d}|{:d}: ".format(epoch, iteration) +
                                             ' '.join(
                                                 '{}={:.2e}'.format(k, loss_components[k]) for k in loss_components.keys()))
                    if logger is not None:
                        loss_tags = list(loss_components.keys())
                        logger.log_training([loss_components[i] for i in loss_tags],
                                            loss_tags, iteration)

                    if (iteration % iters_per_eval) == 0:
                        validator(model, iteration)
                        checkpoint_dict = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "lr": lr,
                            "iteration": iteration,
                            "epoch": epoch
                        }
                        torch.save(checkpoint_dict, join(output_path, "checkpoint_latest.pt"))
                        if (iteration % iters_per_checkpoint) == 0 and iteration > 0:
                            torch.save(checkpoint_dict, join(output_path, "checkpoint_%d.pt" % iteration))
                    iteration += 1
            if rank == 0:
                print(scheduler.get_last_lr())

        except KeyboardInterrupt:
            if rank == 0:
                checkpoint_dict = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "lr": lr,
                    "iteration": iteration,
                    "epoch": epoch
                }
                torch.save(checkpoint_dict, join(output_path, "checkpoint_%d.pt" % iteration))
            sys.exit(0)

    print("Finished!")


class Validator(object):
    def __init__(self, logger: DataLogger, test_file, noise_file,
                 pretrain=True, snr: tuple = None, start_iteration=0):
        self.logger = logger
        self.snr = snr
        self.pretrain = pretrain
        self.augmentation = True
        if pretrain:
            self.augmentation = False

        stft = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=100)
        sample_rate, audio_clean = wavfile.read(test_file)
        print("Eval speech sample rate:", sample_rate)
        sample_rate, noise = wavfile.read(noise_file)
        print("Eval noise sample rate:", sample_rate)
        self.sample_rate = sample_rate
        audio_clean = audio_clean / 32768.0
        audio_clean = audio_clean - np.mean(audio_clean)
        # audio_clean, _, _ = AudioAugmentDataset.scale_db(audio_clean, -25)
        self.audio_clean = audio_clean
        if self.augmentation:
            noise = noise / (max(abs(noise)) + 1)
            if len(noise) < len(audio_clean):
                noise = np.pad(noise, (0, len(audio_clean) - len(noise)), mode='wrap')
            else:
                noise = noise[:len(audio_clean)]
            clean_db = 10 * np.log10(np.mean(audio_clean ** 2) + 1e-9)
            noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-9)
            self.audio_input = []
            for _snr in self.snr:
                noise_scale = np.sqrt(10 ** ((clean_db - noise_db - _snr) / 10)) * noise
                audio_noise = torch.from_numpy(noise_scale + audio_clean).float()
                self.audio_input.append(audio_noise.cuda().unsqueeze(0))
                s_mix = stft(audio_noise).squeeze().numpy()
                # print(np.max(s_mix))
                s_mix_fig = plt.figure(dpi=150, figsize=(9, 3))
                plt.imshow(0.5 * np.log10(s_mix + 1e-12), aspect='auto', origin='lower')
                plt.colorbar()
                self.logger.add_figure("Spectrogram/Noisy_%d_dB" % _snr, s_mix_fig, start_iteration)
                plt.close()
                self.logger.add_audio("Audio/Noisy_%d_dB" % _snr,
                                      (audio_noise/torch.max(torch.abs(audio_noise))).numpy(),
                                      start_iteration,
                                      sample_rate=16000)
            self.audio_input = torch.cat(self.audio_input, dim=0)
        else:
            self.audio_input = torch.from_numpy(audio_clean).float().unsqueeze(0).cuda()
        # Plot audio clean spectrogram to Tensorboard
        s_clean = stft(torch.from_numpy(audio_clean).float().unsqueeze(0)).squeeze().numpy()
        s_clean_fig = plt.figure(dpi=150, figsize=(9, 3))
        plt.imshow(0.5*np.log10(s_clean + 1e-9), aspect='auto', origin='lower')
        plt.colorbar()
        self.logger.add_figure("Spectrogram/Clean", s_clean_fig, start_iteration)
        self.logger.add_audio("Audio/Clean", audio_clean, start_iteration, sample_rate=16000)

    def __call__(self, model, iteration):
        model.eval()
        with torch.no_grad():
            if self.pretrain:
                log_var_speech = model.inference(self.audio_input, pretrain=True)
                log_var_speech = log_var_speech.squeeze().detach().cpu().numpy()
                log_var_speech_fig = plt.figure(dpi=150, figsize=(9, 3))
                plt.imshow(log_var_speech, aspect='auto', origin='lower')
                plt.colorbar()
                self.logger.add_figure("Speech_log_var",
                                       log_var_speech_fig,
                                       iteration)
                plt.close()
            else:
                # x_enhanced, s_noise, s_enhance = model.inference(self.audio_input, self.pretrain)
                # x_enhanced = (x_enhanced / (torch.max(torch.abs(x_enhanced)))).cpu().numpy()
                # s_enhance = s_enhance.detach().cpu().numpy()
                # pesq_dict = {}
                # for i, _snr in enumerate(self.snr):
                #     pesq_dict[str(_snr) + "dB"] = pesq(16000,
                #                                        self.audio_clean[:x_enhanced[i].shape[-1]],
                #                                        x_enhanced[i],
                #                                        "nb")
                #     self.logger.add_audio("Audio/Enhanced_%d_dB" % _snr,
                #                           x_enhanced[i],
                #                           iteration,
                #                           sample_rate=self.sample_rate)
                #     s_enhance_fig = plt.figure(dpi=150, figsize=(9, 3))
                #     plt.imshow(0.5*np.log10(s_enhance[i] + 1e-9), aspect='auto', origin='lower')
                #     plt.colorbar()
                #     self.logger.add_figure("Spectrogram/Enhanced_%d_dB" % _snr,
                #                            s_enhance_fig,
                #                            iteration)
                #     plt.close()
                # print(pesq_dict)
                # self.logger.add_scalars("PESQ", pesq_dict, iteration)
                # audio_clean = torch.from_numpy(self.audio_clean).float().unsqueeze(0).cuda()
                # audio_clean = audio_clean.repeat(self.audio_input.shape[0], 1)
                x_enhance, s_enhance, log_var_speech, log_var_noise = model.inference(self.audio_input,
                                                                                      pretrain=False)
                x_enhance = (x_enhance / (torch.max(torch.abs(x_enhance)))).cpu().numpy()
                s_enhance = s_enhance.detach().cpu().numpy()
                log_var_noise = log_var_noise.detach().cpu().numpy()
                log_var_speech = log_var_speech.detach().cpu().numpy()
                pesq_dict = {}
                for i, _snr in enumerate(self.snr):
                    self.logger.add_audio("Audio/Enhanced_%d_dB" % _snr,
                                          x_enhance[i],
                                          iteration,
                                          sample_rate=self.sample_rate)

                    pesq_dict[str(_snr) + "dB"] = pesq(16000,
                                                       self.audio_clean[:x_enhance[i].shape[-1]],
                                                       x_enhance[i],
                                                       "nb")

                    s_enhance_fig = plt.figure(dpi=150, figsize=(9, 3))
                    plt.imshow(0.5*np.log10(s_enhance[i] + 1e-9), aspect='auto', origin='lower')
                    plt.colorbar()
                    self.logger.add_figure("Spectrogram/Enhanced_%d_dB" % _snr,
                                           s_enhance_fig,
                                           iteration)
                    plt.close()

                    log_var_speech_fig = plt.figure(dpi=150, figsize=(9, 3))
                    plt.imshow(log_var_speech[i], aspect='auto', origin='lower')
                    plt.colorbar()
                    self.logger.add_figure("Speech_log_var/%d_dB" % _snr,
                                           log_var_speech_fig,
                                           iteration)
                    plt.close()

                    log_var_noise_fig = plt.figure(dpi=150, figsize=(9, 3))
                    plt.imshow(log_var_noise[i], aspect='auto', origin='lower')
                    plt.colorbar()
                    self.logger.add_figure("Noise_log_var/%d_dB" % _snr,
                                           log_var_noise_fig,
                                           iteration)
                    plt.close()
                print(pesq_dict)
                self.logger.add_scalars("PESQ", pesq_dict, iteration)
        model.train()


# Example:
# python train.py --config=cfg/train_config_cvq.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    global args
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    with open(args.config) as f:
        config = json.load(f)
    training_configs = config["training_configs"]

    global dist_configs
    dist_configs = config["dist_configs"]
    global model_configs
    model_configs = config["model_configs"]
    global validation_configs
    validation_configs = config["validation_configs"]
    global data_configs
    data_configs = config["data_configs"]

    if num_gpus > 1:
        if args.group_name == '':
            print("Warning: Training on 1 GPU!")
            num_gpus = 1
        else:
            print("Run distributed training on %d GPUs" % num_gpus)
    train(num_gpus=num_gpus, rank=args.rank, group_name=args.group_name, **training_configs)
