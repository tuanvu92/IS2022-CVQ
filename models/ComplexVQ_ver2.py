import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
from models.conv_stft import *
import time
import json
import os
from baseline.show import show_params, show_model
import torch.fft


class ChannelAttention(nn.Module):
    def __init__(self, input_dim, n_lookback=5):
        super().__init__()
        self.n_lookback = n_lookback
        self.q_layer = nn.ModuleList()
        self.k_layer = nn.ModuleList()
        for i in range(n_lookback):
            self.q_layer.append(ConvNorm(input_dim, input_dim, kernel_size=1))
            self.k_layer.append(ConvNorm(input_dim, input_dim, kernel_size=1))
        self.v_layer = nn.Sequential(
            ConvNorm(input_dim, input_dim, kernel_size=1),
            nn.BatchNorm1d(input_dim, momentum=0.8),
            nn.PReLU()
        )

    def forward(self, x):
        q = []
        k = []
        for i in range(self.n_lookback):
            _q = self.q_layer[i](x)
            _k = self.k_layer[i](x)
            if i > 0:
                _q = F.pad(_q, [i, 0])[..., :-i]
                _k = F.pad(_k, [i, 0])[..., :-i]
            q.append(_q)
            k.append(_k)
        # q.shape = [B, T, C, d]
        q = torch.stack(q, dim=-1).permute(0, 2, 1, 3)
        k = torch.stack(k, dim=-1).permute(0, 2, 1, 3)
        # print(q.shape, k.shape)
        # w.shape = [B, C, T], v.shape = [B, C, T]
        w = torch.softmax(torch.sum(k @ q.permute(0, 1, 3, 2), dim=-1), dim=-1).permute(0, 2, 1)
        v = self.v_layer(x)
        output = v * w
        return output


class PhaseEstimatorRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dilation_rate=None, complex_axis=0, pad_mode='causal'):
        super().__init__()
        self.complex_axis = complex_axis
        self.input_layer = nn.Sequential(
            ComplexConv1d(input_dim, hidden_dim, kernel_size=5, pad_mode=pad_mode),
            ComplexBatchNorm(hidden_dim, momentum=0.8),
            nn.PReLU()
        )
        self.WN = nn.ModuleList()
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 1, 2, 4]
        for d in dilation_rate:
            self.WN.append(ComplexWNCell(residual_dim=hidden_dim,
                                         gate_dim=hidden_dim,
                                         skip_dim=hidden_dim,
                                         dilation=d,
                                         pad_mode=pad_mode))

        self.rnn_r = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
        self.rnn_i = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)

        self.output_layer = nn.Sequential(
            ComplexConv1d(hidden_dim, input_dim, kernel_size=3, pad_mode=pad_mode),
            ComplexBatchNorm(input_dim, momentum=0.25),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.input_layer(x)
        skip = 0
        for i in range(len(self.WN)):
            h, _skip = self.WN[i](h)
            skip += _skip
        skip *= math.sqrt(1.0 / len(self.WN))
        h_r, _ = self.rnn_r(skip.transpose(1, 2))
        h_r = h_r.transpose(1, 2)
        h_i, _ = self.rnn_i(skip.transpose(1, 2))
        h_i = h_i.transpose(1, 2)

        h_rr, h_ir = torch.chunk(h_r, 2, self.complex_axis)
        h_ri, h_ii = torch.chunk(h_i, 2, self.complex_axis)
        h_r = h_rr - h_ii
        h_i = h_ri + h_ir
        h = torch.cat([h_r, h_i], self.complex_axis)
        mask = self.output_layer(h)
        return mask


class GainPredictor(nn.Module):
    def __init__(self, input_dim, pad_mode='causal'):
        super(GainPredictor, self).__init__()
        self.nn = Decoder(input_dim=input_dim,
                          output_dim=1,
                          residual_dim=32, gate_dim=32,
                          skip_dim=32, n_stage=2, kernel_size=3,
                          cond_dim=input_dim, n_upsample_factor=0, pad_mode=pad_mode,
                          dilation_rate=[1, 2, 4])

    def forward(self, s_abs_2, cond):
        return self.nn(torch.log(s_abs_2 + 1e-12), cond)


class VQUnet(nn.Module):
    def __init__(self, input_dim,
                 quantize_configs=None,
                 encoder_configs=None,
                 decoder_configs=None,
                 pad_mode='causal'):
        super().__init__()
        self.encoder_bot = Encoder(input_dim=input_dim, pad_mode=pad_mode, **encoder_configs["bot"])

        self.encoder_top = Encoder(input_dim=encoder_configs["bot"]["output_dim"],
                                   output_dim=quantize_configs["top"]["emb_dim"],
                                   pad_mode=pad_mode,
                                   **encoder_configs["top"])

        # ===== For noisy input =====
        self.encoder_bot_noise = Encoder(input_dim=input_dim, pad_mode=pad_mode, **encoder_configs["bot"])

        self.encoder_top_noise = Encoder(input_dim=encoder_configs["bot"]["output_dim"],
                                         output_dim=quantize_configs["top"]["emb_dim"],
                                         pad_mode=pad_mode,
                                         **encoder_configs["top"])
        self.quantize_conv_bot_noise = ConvNorm(in_channels=encoder_configs["bot"]["output_dim"],
                                                out_channels=quantize_configs["bot"]["emb_dim"],
                                                kernel_size=3, pad_mode=pad_mode)
        # ====== END =====

        self.encoder_cepstrum = Encoder(input_dim=input_dim, output_dim=2, down_sample_factor=0,
                                        pad_mode=pad_mode, residual_dim=64, gate_dim=64,
                                        skip_dim=32, kernel_size=3,
                                        n_stage=2, dilation_rate=[1, 2, 4])

        self.decoder_top = Decoder(input_dim=quantize_configs["top"]["emb_dim"],
                                   pad_mode=pad_mode,
                                   **decoder_configs["top"])

        self.decoder_bot = Decoder(input_dim=quantize_configs["bot"]["emb_dim"] + decoder_configs["top"]["output_dim"],
                                   output_dim=input_dim,
                                   cond_dim=2,
                                   pad_mode=pad_mode,
                                   **decoder_configs["bot"])

        self.quantize_conv_bot = ConvNorm(in_channels=encoder_configs["bot"]["output_dim"],
                                          out_channels=quantize_configs["bot"]["emb_dim"],
                                          kernel_size=3, pad_mode=pad_mode)

        self.quantize_top = VectorQuantize(**quantize_configs["top"])
        self.quantize_bot = VectorQuantize(**quantize_configs["bot"])

    def forward(self, s_abs_2_inputs, train_clean=True):
        if train_clean:
            return self.train_clean(s_abs_2_inputs[0])
        else:
            return self.train_noisy(s_abs_2_inputs[0], s_abs_2_inputs[1])

    def train_clean(self, s_mix_abs_2):
        """ Calculate forward pass
                Args:
                    s_mix_abs_2 (Tensor): energy STFT spectrum of mixture speech

                Returns:
                    log_var_clean (Tensor): estimated speech variance from clean speech
                    log_var_mix (Tensor): estimated speech variance from noisy speech
                """
        # batch_size = s_mix_abs_2.shape[0]
        c_mix = self.sp2cep(s_mix_abs_2)
        h_ceps = self.encoder_cepstrum(c_mix)
        z, z_id, encoder_loss, perplexity = self.encode_train(torch.log(s_mix_abs_2+1e-12))
        log_var = self.decode(z, h_ceps)
        vq_loss, commitment_loss = encoder_loss
        # loss = vq_loss + 0.25 * commitment_loss + rc_loss_clean + rc_loss_mix_1 + rc_loss_mix_2
        return log_var, vq_loss, commitment_loss, perplexity

    def train_noisy(self, s_clean_abs_2, s_mix_abs_2):
        c_mix = self.sp2cep(s_mix_abs_2)
        h_ceps = self.encoder_cepstrum(c_mix)
        z, z_id, encoder_loss, perplexity = self.encode_train_noisy(torch.log(s_clean_abs_2+1e-12),
                                                                    torch.log(s_mix_abs_2+1e-12))
        log_var = self.decode(z, h_ceps)
        vq_loss, commitment_loss = encoder_loss
        # loss = vq_loss + 0.25 * commitment_loss + rc_loss_clean + rc_loss_mix_1 + rc_loss_mix_2
        return log_var, vq_loss, commitment_loss, perplexity

    def inference(self, s_abs_2):
        c_mix = self.sp2cep(s_abs_2)
        h_ceps = self.encoder_cepstrum(c_mix)
        z, z_id = self.encode_inference(torch.log(s_abs_2+1e-12))
        log_var = self.decode(z, h_ceps)
        return log_var

    def encode_train(self, x_mix):
        """ Encode stft spectrum into discrete latent
        Args:
            x_mix (Tensor): input tensor of mixture speech spectral

        Returns:
            Lists of Tensor: Latent variables and loss components (if return_loss is True)

        """
        h_bot = self.encoder_bot(x_mix)
        h_top = self.encoder_top(h_bot)
        h_bot_q = self.quantize_conv_bot(h_bot)

        z_top, z_id_top = self.quantize_top(h_top)
        z_bot, z_id_bot = self.quantize_bot(h_bot_q)

        z_top_st = h_top + (z_top - h_top).detach()
        z_bot_st = h_bot_q + (z_bot - h_bot_q).detach()

        vq_loss = (h_top - z_top.detach()).pow(2).mean() + \
                  (h_bot_q - z_bot.detach()).pow(2).mean()

        commitment_loss = (h_top.detach() - z_top).pow(2).mean() + \
                          (h_bot_q.detach() - z_bot).pow(2).mean()

        perplexity_top = self.calculate_perplexity(z_id_top, self.quantize_top.n_emb)
        perplexity_bot = self.calculate_perplexity(z_id_bot, self.quantize_bot.n_emb)

        return [z_top_st, z_bot_st], \
               [z_id_top.detach(), z_id_bot.detach()], \
               [vq_loss, commitment_loss], \
               [perplexity_top, perplexity_bot]

    def encode_train_noisy(self, x_clean, x_mix):
        """ Encode stft spectrum into discrete latent
        Args:
            x_mix (Tensor): input tensor of mixture speech spectral

        Returns:
            Lists of Tensor: Latent variables and loss components (if return_loss is True)

        """

        h_bot_clean = self.encoder_bot(x_clean)
        h_top_clean = self.encoder_top(h_bot_clean)
        h_bot_q_clean = self.quantize_conv_bot(h_bot_clean)

        z_top_clean, z_id_top_clean = self.quantize_top(h_top_clean)
        z_bot_clean, z_id_bot_clean = self.quantize_bot(h_bot_q_clean)

        h_bot_mix = self.encoder_bot_noise(x_mix)
        h_top_mix = self.encoder_top_noise(h_bot_mix)
        h_bot_q_mix = self.quantize_conv_bot_noise(h_bot_mix)

        z_top_mix, z_id_top_mix = self.quantize_top(h_top_mix)
        z_bot_mix, z_id_bot_mix = self.quantize_bot(h_bot_q_mix)

        # z_top_st = h_top_mix + (z_top_mix - h_top_mix).detach()
        # z_bot_st = h_bot_q_mix + (z_bot_mix - h_bot_q_mix).detach()

        vq_loss = (h_top_mix - z_top_clean.detach()).pow(2).mean() + \
                  (h_bot_q_mix - z_bot_clean.detach()).pow(2).mean()

        # commitment_loss = (h_top[:batch_size].detach() - z_top[:batch_size]).pow(2).mean() + \
        #                   (h_bot_q[:batch_size].detach() - z_bot[:batch_size]).pow(2).mean()
        commitment_loss = torch.tensor(0.).cuda()
        perplexity_top = self.calculate_perplexity(z_id_top_mix, self.quantize_top.n_emb)
        perplexity_bot = self.calculate_perplexity(z_id_bot_mix, self.quantize_bot.n_emb)

        return [z_top_mix.detach(), z_bot_mix.detach()], \
               [z_id_top_mix.detach(), z_id_bot_mix.detach()], \
               [vq_loss, commitment_loss], \
               [perplexity_top, perplexity_bot]

    def encode_inference(self, x_mix):
        """ Encode stft spectrum into discrete latent
        Args:
            x_mix (Tensor): input tensor of mixture speech spectral

        Returns:
            Lists of Tensor: Latent variables

        """
        h_bot = self.encoder_bot_noise(x_mix)
        h_top = self.encoder_top_noise(h_bot)

        # h_bot = self.ca_bot(h_bot)
        # h_top = self.ca_top(h_top)

        h_bot_q = self.quantize_conv_bot_noise(h_bot)

        z_top, z_id_top = self.quantize_top(h_top)
        z_bot, z_id_bot = self.quantize_bot(h_bot_q)

        return [z_top, z_bot], [z_id_top, z_id_bot]

    def decode(self, z, cond):
        """ Calculate decode pass
        Args:
            z (Tensor): latent variable

        Returns:
            Tensor: output tensor

        """
        z_top, z_bot = z
        z_top_dec = self.decoder_top(z_top)
        x_hat = self.decoder_bot(torch.cat([z_bot, z_top_dec], dim=1), cond)
        return x_hat

    @staticmethod
    def calculate_perplexity(z_id, codebook_size):
        z_id = z_id.flatten()
        z_id_onehot = torch.eye(codebook_size, dtype=torch.float32).cuda().index_select(dim=0, index=z_id)
        avg_probs = z_id_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    @staticmethod
    def sp2cep(sp):
        sp = torch.log(sp + 1e-12)
        if len(sp.shape) == 2:
            sp = sp.unsqueeze(0)
        n = sp.shape[1]
        # sp = sp.transpose(1, 2).unsqueeze(-1)
        # sp = torch.cat([sp, torch.zeros_like(sp)], dim=-1)
        # c = torch.irfft(sp, signal_ndim=1, signal_sizes=[2 * (n - 1)]).transpose(1, 2)[:, :n]
        # print("2:", sp.shape)
        c = torch.fft.irfft(sp, n=2 * (n - 1), dim=1)[:, :n]
        c[:, 0] /= 2.
        return c

    @staticmethod
    def cep2sp(c):
        if len(c.shape) == 2:
            c = c.unsqueeze(0)
        c[:, 0] *= 2.
        # c = c.transpose(1, 2)
        c = torch.cat([c, torch.flip(c[:, 1:-1], dims=[1])], dim=1)
        sp = torch.fft.rfft(c, dim=1).real
        # sp = torch.exp(sp)
        return torch.exp(sp.transpose(1, 2))


class NoiseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_configs, decoder_configs, pad_mode='causal'):
        super().__init__()
        self.latent_dim = latent_dim

        self.log_var_speech_encoder = Encoder(input_dim=input_dim, output_dim=16,
                                              residual_dim=64, gate_dim=64, skip_dim=64,
                                              down_sample_factor=0,
                                              kernel_size=3,
                                              pad_mode=pad_mode)
        # self.encoder = Encoder(input_dim=input_dim, cond_dim=16,
        #                        pad_mode=pad_mode, **encoder_configs)
        # self.z_mean_logvar_layer = ConvNorm(encoder_configs["output_dim"],
        #                                     2*latent_dim,
        #                                     kernel_size=1)
        self.decoder = Decoder(
            input_dim=input_dim,
            output_dim=input_dim,
            cond_dim=16,
            pad_mode=pad_mode,
            **decoder_configs
        )

    def forward(self, x, log_var_speech):
        h_speech = self.log_var_speech_encoder(log_var_speech)
        noise_logvar = self.decoder(x, h_speech)
        return noise_logvar

    @staticmethod
    def sampling(z_mean, z_logvar):
        z = z_mean + torch.exp(z_logvar) * torch.randn_like(z_logvar)
        return z


class ComplexVQ2(nn.Module):
    """ Speech enhancement with VQUnet and phase correction """
    def __init__(self,
                 stft_configs=None,
                 vq_configs=None,
                 noise_vae_configs=None,
                 phase_estimator_configs=None):
        """ Initialized VQ-VAE modules
        Args:
            stft_configs: Configuration for STFT
            vq_configs: Configuration for VQ-UNET
            noise_vae_configs: configuration for noise VAE
            phase_estimator_configs: configuration for phase correction net
        """
        super().__init__()
        self.look_ahead_frames = 6
        assert stft_configs is not None
        assert vq_configs is not None
        assert noise_vae_configs is not None
        assert phase_estimator_configs is not None
        # STFT - iSTFT layers
        self.stft = ConvSTFT(**stft_configs)
        self.istft = ConviSTFT(**stft_configs)
        self.n_features = stft_configs["fft_len"]//2 + 1
        # Speech VQ-UNET
        self.vq_unet = VQUnet(input_dim=self.n_features,
                              **vq_configs)
        # self.gain_predictor = GainPredictor(input_dim=self.n_features)
        # Noise VAE
        self.noise_vae = NoiseVAE(input_dim=self.n_features,
                                  **noise_vae_configs)
        # Phase estimator
        self.phase_estimator = PhaseEstimatorRNN(input_dim=self.n_features,
                                                 **phase_estimator_configs)

    def forward(self, x, train_clean=True):
        if train_clean:
            return self.train_clean(x[0])
        else:
            return self.train_noisy(x[0], x[1])

    def train_noisy(self, x_clean, x_mix):
        """ Calculate loss for training
        Args:
            x_clean (Tensor): clean speech waveform
            x_mix (Tensor): noisy speech waveform

        """
        with torch.no_grad():
            # Remove DC
            x_noise = x_mix - x_clean
            # x_clean = x_clean - torch.mean(x_clean, dim=-1, keepdim=True)
            # x_mix = x_mix - torch.mean(x_mix, dim=-1, keepdim=True)

            s_mix = self.stft(x_mix)
            s_clean = self.stft(x_clean)
            s_noise = self.stft(x_noise)

            s_clean = s_clean[:, :, :8*(s_clean.shape[-1]//8)]
            s_mix = s_mix[:, :, :8*(s_mix.shape[-1] // 8)]
            s_noise = s_noise[..., :8*(s_noise.shape[-1] // 8)]
            batch_size = s_mix.shape[0]
            n_fft = s_mix.shape[1] // 2
            s_mix_r, s_mix_i = s_mix[:, :n_fft, :], s_mix[:, n_fft:, :]
            s_clean_r, s_clean_i = s_clean[:, :n_fft, :], s_clean[:, n_fft:, :]
            s_noise_r, s_noise_i = s_noise[:, :n_fft, :], s_noise[:, n_fft:, :]

            s_mix_abs_2 = s_mix_r**2 + s_mix_i**2
            s_clean_abs_2 = s_clean_r**2 + s_clean_i**2
            s_noise_abs_2 = s_noise_r**2 + s_noise_i**2

        log_var_speech, vq_loss, commitment_loss, perplexity = self.vq_unet([s_clean_abs_2, s_mix_abs_2],
                                                                            train_clean=False)
        log_var_speech = log_var_speech[..., self.look_ahead_frames:]
        s_mix_abs_2 = s_mix_abs_2[..., :-self.look_ahead_frames]
        s_noise_abs_2 = s_noise_abs_2[..., :-self.look_ahead_frames]
        s_clean_abs_2 = s_clean_abs_2[..., :-self.look_ahead_frames]
        rc_loss_clean = (log_var_speech - torch.log(s_clean_abs_2 + 1e-12) +
                         s_clean_abs_2 * torch.exp(-log_var_speech)).mean()

        log_var_noise = self.noise_vae(torch.log(s_mix_abs_2 + 1e-12) - log_var_speech.detach(),
                                       log_var_speech.detach())
        log_var_noise = log_var_noise[..., self.look_ahead_frames:]
        log_var_speech = log_var_speech[..., :log_var_noise.shape[-1]]

        # gain = self.gain_predictor(s_mix_abs_2[..., :log_var_noise.shape[-1]],
        #                            log_var_speech - log_var_noise)
        #
        # gain = gain[..., self.look_ahead_frames:]
        # log_var_noise = log_var_noise[..., :gain.shape[-1]]
        # log_var_speech = log_var_speech[..., :gain.shape[-1]]
        # s_mix_abs_2 = s_mix_abs_2[..., :gain.shape[-1]]
        #
        # log_var_mix = torch.log(torch.exp(log_var_speech.detach() + gain) + torch.exp(log_var_noise.detach()))
        # rc_loss_mix = (log_var_mix - torch.log(s_mix_abs_2 + 1e-12) + s_mix_abs_2 * torch.exp(-log_var_mix)).mean()

        rc_loss_noise = (log_var_noise - torch.log(s_noise_abs_2[..., :log_var_noise.shape[-1]] + 1e-12)
                         + s_noise_abs_2[..., :log_var_noise.shape[-1]] * torch.exp(-log_var_noise)).mean()

        tf_gain = torch.exp(log_var_speech) / (torch.exp(log_var_speech) + torch.exp(log_var_noise))
        # Log-spectral distortion loss
        s_mix_abs_2 = s_mix_abs_2[..., :tf_gain.shape[-1]]
        s_clean_abs_2 = s_clean_abs_2[..., :tf_gain.shape[-1]]
        s_enhance_abs_2 = tf_gain * s_mix_abs_2
        noise_pred_abs_2 = (1-tf_gain) * s_mix_abs_2
        lsd_speech = (torch.log(s_enhance_abs_2 + 1e-12) - torch.log(s_clean_abs_2 + 1e-12)).pow(2).mean()
        # lsd_noise = (torch.log(noise_pred_abs_2 + 1e-12) - torch.log(s_noise_abs_2 + 1e-12)).pow(2).mean()
        with torch.no_grad():
            s_enhance_noisy_phase = torch.cat([s_mix_r[..., :tf_gain.shape[-1]]*torch.sqrt(tf_gain),
                                               s_mix_i[..., :tf_gain.shape[-1]]*torch.sqrt(tf_gain)],
                                              dim=0)
        mask = self.phase_estimator(s_enhance_noisy_phase.detach())
        mask = mask[..., self.look_ahead_frames:self.look_ahead_frames+tf_gain.shape[-1]]
        mask_r, mask_i = mask[:batch_size], mask[batch_size:]

        s_enhance_phase = self.complex_masking(mask_r, mask_i,
                                               s_mix_r[..., :mask.shape[-1]].detach(),
                                               s_mix_i[..., :mask.shape[-1]].detach(),
                                               tf_gain[..., :mask.shape[-1]])

        x_enhance = self.istft(s_enhance_phase).squeeze(dim=1)
        x_clean = x_clean[:, :x_enhance.shape[1]]
        snr_loss = -self.si_snr(x_enhance, x_clean)

        loss = snr_loss + rc_loss_noise + rc_loss_clean + vq_loss + 0.25*commitment_loss
        return {"loss": loss,
                "snr": -snr_loss,
                # "lsd": lsd,
                # "rc_mix": rc_loss_mix,
                "rc_clean": rc_loss_clean,
                "rc_noise": rc_loss_noise,
                "vql": vq_loss,
                "plx_top": perplexity[0],
                "plx_bot": perplexity[1]}
        # loss = vq_loss + rc_loss_clean
        # return {"loss": loss,
        #         "rc_clean": rc_loss_clean,
        #         "vql": vq_loss,
        #         "plx_top": perplexity[0],
        #         "plx_bot": perplexity[1]}

    def train_clean(self, x_clean):
        """ Calculate loss for training
        Args:
            x_clean (Tensor): clean speech waveform

        """
        with torch.no_grad():
            # Remove DC
            x_clean = x_clean - torch.mean(x_clean, dim=-1, keepdim=True)
            # print("0:", x_clean.shape)
            s_clean = self.stft(x_clean)
            # print("1:", s_clean.shape)
            s_clean = s_clean[:, :, :8*(s_clean.shape[-1]//8)]
            n_fft = s_clean.shape[1] // 2
            s_clean_r, s_clean_i = s_clean[:, :n_fft, :], s_clean[:, n_fft:, :]
            s_clean_abs_2 = s_clean_r**2 + s_clean_i**2

        log_var_hat, vq_loss, commitment_loss, perplexity = self.vq_unet([s_clean_abs_2], train_clean=True)
        rc_loss_clean = self.is_divergence(s_clean_abs_2, log_var_hat).mean()
        loss = rc_loss_clean + vq_loss + 0.25*commitment_loss  # + snr_loss
        return {"loss": loss,
                "rc_clean": rc_loss_clean,
                "vql": vq_loss,
                "plx_top": perplexity[0],
                "plx_bot": perplexity[1]}

    def si_snr(self, s1, s2, alpha=1., eps=1e-9):
        s1_s2_norm = self.l2_norm(s1, s2)
        s2_s2_norm = self.l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
        e_noise = s1 - s_target
        target_norm = self.l2_norm(s_target, s_target)
        noise_norm = self.l2_norm(e_noise, e_noise)
        snr = 10 * torch.log10(target_norm/(noise_norm + eps) + eps)
        return torch.mean(alpha*snr)

    def is_divergence(self, x_abs_2, log_var):
        x_abs_2 = x_abs_2[:, :, :-self.look_ahead_frames]
        log_var = log_var[:, :, self.look_ahead_frames:]
        log_x_abs_2 = torch.log(x_abs_2 + 1e-12)
        loss = log_var - log_x_abs_2 + x_abs_2 * torch.exp(-log_var)
        return loss

    @staticmethod
    def l2_norm(s1, s2):
        norm = torch.sum(s1 * s2, -1, keepdim=True)
        return norm

    @staticmethod
    def sp2cep(sp):
        sp = torch.log(sp + 1e-12)
        if len(sp.shape) == 2:
            sp = sp.unsqueeze(0)
        n = sp.shape[1]
        # sp = sp.transpose(1, 2).unsqueeze(-1)
        # sp = torch.cat([sp, torch.zeros_like(sp)], dim=-1)
        # c = torch.irfft(sp, signal_ndim=1, signal_sizes=[2 * (n - 1)]).transpose(1, 2)[:, :n]
        c = torch.fft.irfft(sp, n=2 * (n - 1), dim=1)[:, :n]
        c[:, 0] /= 2.
        return c

    @staticmethod
    def cep2sp(c):
        if len(c.shape) == 2:
            c = c.unsqueeze(0)
        c[:, 0] *= 2.
        # c = c.transpose(1, 2)
        c = torch.cat([c, torch.flip(c[:, 1:-1], dims=[1])], dim=1)
        sp = torch.fft.rfft(c, dim=1).real
        # sp = torch.exp(sp)
        return torch.exp(sp.transpose(1, 2))

    def complex_masking(self, m_r, m_i, s_r, s_i, gain, eps=1e-9):
        """ Apply complex masking for phase correction
        :param m_r: mask real part
        :param m_i: mask imaginary part
        :param s_r: real spectrum
        :param s_i: imaginary spectrum
        :param gain: magnitude gain
        :param eps: small value to prevent overflow
        :return: filtered complex spectrum (complex_axis = 0)
        """
        m_mags = (m_r ** 2 + m_i ** 2) ** 0.5
        m_phase = torch.atan2(m_i / (m_mags + eps), m_r / (m_mags + eps))
        s_noise_mags = (s_r ** 2 + s_i ** 2) ** 0.5
        s_noise_phase = torch.atan2(s_i / (s_noise_mags + eps), s_r / (s_noise_mags + eps))
        s_hat_phase = s_noise_phase + m_phase
        s_hat_mags = s_noise_mags * gain
        s_hat_r = s_hat_mags * torch.cos(s_hat_phase)
        s_hat_i = s_hat_mags * torch.sin(s_hat_phase)
        s_hat = torch.cat([s_hat_r, s_hat_i], dim=1)
        return s_hat

    def inference(self, x_mix, pretrain=True):
        with torch.no_grad():
            # Remove DC
            x_mix = x_mix - torch.mean(x_mix, dim=-1, keepdim=True)
            # x_clean = x_clean - torch.mean(x_clean, dim=-1, keepdim=True)
            s_mix = self.stft(x_mix)
            # s_clean = self.stft(x_clean)
            s_mix = s_mix[..., :8*(s_mix.shape[-1]//8)]
            # s_clean = s_clean[..., :8*(s_clean.shape[-1]//8)]
            batch_size = s_mix.shape[0]
            n_fft = s_mix.shape[1] // 2
            s_mix_r, s_mix_i = s_mix[:, :n_fft], s_mix[:, n_fft:]
            # s_clean_r, s_clean_i = s_clean[:, :n_fft], s_clean[:, n_fft:]
            # s_mix = torch.cat([s_mix_r, s_mix_i], dim=0)
            # s_clean = torch.cat([s_clean_r, s_clean_i], dim=0)
            s_mix_abs_2 = s_mix_r ** 2 + s_mix_i ** 2
            # s_clean_abs_2 = s_clean_r ** 2 + s_clean_i ** 2

            log_var_speech = self.vq_unet.inference(s_mix_abs_2)

            if pretrain:
                return log_var_speech
            else:
                log_var_speech = log_var_speech[..., self.look_ahead_frames:]
                s_mix_abs_2 = s_mix_abs_2[..., :-self.look_ahead_frames]
                log_var_noise = self.noise_vae(torch.log(s_mix_abs_2 + 1e-12) - log_var_speech.detach(),
                                               log_var_speech.detach())
                log_var_noise = log_var_noise[..., self.look_ahead_frames:]
                log_var_speech = log_var_speech[..., :log_var_noise.shape[-1]]
                tf_gain = torch.exp(log_var_speech)/(torch.exp(log_var_speech) + torch.exp(log_var_noise))
                s_enhance_noisy_phase = torch.cat([s_mix_r[..., :tf_gain.shape[-1]] * torch.sqrt(tf_gain),
                                                   s_mix_i[..., :tf_gain.shape[-1]] * torch.sqrt(tf_gain)],
                                                  dim=0)
                mask = self.phase_estimator(s_enhance_noisy_phase)
                mask = mask[..., self.look_ahead_frames:self.look_ahead_frames + tf_gain.shape[-1]]
                mask_r, mask_i = mask[:batch_size], mask[batch_size:]

                s_enhance_phase = self.complex_masking(mask_r, mask_i,
                                                       s_mix_r[..., :mask.shape[-1]].detach(),
                                                       s_mix_i[..., :mask.shape[-1]].detach(),
                                                       tf_gain[..., :mask.shape[-1]])

                x_hat = self.istft(s_enhance_phase).squeeze(dim=1)
                s_enhance_abs_2 = tf_gain * s_mix_abs_2[..., :tf_gain.shape[-1]]

                # tf_gain_noise = torch.exp(log_var_noise) / (torch.exp(log_var_speech) + torch.exp(log_var_noise))
                # s_noise_abs_2 = tf_gain_noise * s_mix_abs_2[..., :tf_gain_noise.shape[-1]]

                # snr_pred = 10 * torch.log10(
                #     torch.mean(torch.sum(s_enhance_abs_2, dim=1) / torch.sum(s_mix_abs_2[..., :tf_gain.shape[-1]], dim=1)))
                return x_hat, s_enhance_abs_2, log_var_speech, log_var_noise

    def inference_no_phase(self, x_mix, pretrain=True):
        with torch.no_grad():
            # Remove DC
            x_mix = x_mix - torch.mean(x_mix, dim=-1, keepdim=True)
            # x_clean = x_clean - torch.mean(x_clean, dim=-1, keepdim=True)
            s_mix = self.stft(x_mix)
            # s_clean = self.stft(x_clean)
            s_mix = s_mix[..., :8 * (s_mix.shape[-1] // 8)]
            # s_clean = s_clean[..., :8*(s_clean.shape[-1]//8)]
            batch_size = s_mix.shape[0]
            n_fft = s_mix.shape[1] // 2
            s_mix_r, s_mix_i = s_mix[:, :n_fft], s_mix[:, n_fft:]
            # s_clean_r, s_clean_i = s_clean[:, :n_fft], s_clean[:, n_fft:]
            # s_mix = torch.cat([s_mix_r, s_mix_i], dim=0)
            # s_clean = torch.cat([s_clean_r, s_clean_i], dim=0)
            s_mix_abs_2 = s_mix_r ** 2 + s_mix_i ** 2
            # s_clean_abs_2 = s_clean_r ** 2 + s_clean_i ** 2

            log_var_speech = self.vq_unet.inference(s_mix_abs_2)

            if pretrain:
                return log_var_speech
            else:
                log_var_speech = log_var_speech[..., self.look_ahead_frames:]
                s_mix_abs_2 = s_mix_abs_2[..., :-self.look_ahead_frames]
                log_var_noise = self.noise_vae(torch.log(s_mix_abs_2 + 1e-12) - log_var_speech.detach(),
                                               log_var_speech.detach())
                log_var_noise = log_var_noise[..., self.look_ahead_frames:]
                log_var_speech = log_var_speech[..., :log_var_noise.shape[-1]]
                tf_gain = torch.exp(log_var_speech) / (torch.exp(log_var_speech) + torch.exp(log_var_noise))
                s_enhance_noisy_phase = torch.cat([s_mix_r[..., :tf_gain.shape[-1]] * torch.sqrt(tf_gain),
                                                   s_mix_i[..., :tf_gain.shape[-1]] * torch.sqrt(tf_gain)],
                                                  dim=0)
                s_enhance_noisy_phase = torch.cat([s_enhance_noisy_phase[:batch_size],
                                                   s_enhance_noisy_phase[batch_size:]],
                                                  dim=1)
                x_hat = self.istft(s_enhance_noisy_phase).squeeze(dim=1)
                s_enhance_abs_2 = tf_gain * s_mix_abs_2[..., :tf_gain.shape[-1]]

                tf_gain_noise = torch.exp(log_var_noise) / (torch.exp(log_var_speech) + torch.exp(log_var_noise))
                s_noise_abs_2 = tf_gain_noise * s_mix_abs_2[..., :tf_gain_noise.shape[-1]]

                snr_pred = 10*torch.log10(torch.mean(torch.sum(s_enhance_abs_2, dim=1)/torch.sum(s_noise_abs_2, dim=1)))

                return x_hat, s_enhance_abs_2, log_var_speech, log_var_noise, snr_pred

    def copy_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if (k in model_dict) and (model_dict[k].shape == v.shape):
                model_dict[k] = v
            else:
                print("Ignore mismatch weight: ", k)

            # Copy weight to encoder noise
            # if "vq_unet.encoder_bot" in k:
            #     model_dict[k.replace("encoder_bot", "encoder_bot_noise")] = v
            # if "vq_unet.encoder_top" in k:
            #     model_dict[k.replace("encoder_top", "encoder_top_noise")] = v
            # if "vq_unet.quantize_conv_bot" in k:
            #     model_dict[k.replace("quantize_conv_bot", "quantize_conv_bot_noise")] = v

        self.load_state_dict(model_dict)


if __name__ == "__main__":
    with open("../cfg/train_config_cvq_2.json", "r") as f:
        cfg = json.load(f)
    model = ComplexVQ2(**cfg["model_configs"]).cpu().eval()
    inputs = torch.randn([1, 16000 * 8]).clamp_(-1, 1)
    start = time.time()
    model.inference(inputs)
    end = time.time()
    print(end - start)


