# -*- coding: utf-8 -*-
""" Definition for common layers

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class ConvNorm(nn.Conv1d):
    """ 1D convolution layer with padding mode """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 pad_mode="same", dilation=1, groups=1, bias=True, w_init_gain='linear'):
        self.pad_mode = pad_mode
        self._stride = stride
        if pad_mode == "same":
            _pad = int((dilation * (kernel_size - 1) + 1 - stride) / 2)
        elif pad_mode == "causal":
            _pad = dilation * (kernel_size - 1) - (stride - 1)
        else:
            _pad = 0
        self._pad = _pad
        super(ConvNorm, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=_pad,
                                       dilation=dilation,
                                       bias=bias,
                                       groups=groups)
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        """ Calculate forward propagation
        Args:
            signal (Tensor): input tensor

        Returns:
            Tensor: output tensor

        """
        conv_signal = super(ConvNorm, self).forward(signal)
        if self.pad_mode == "causal":
            if self._pad > 0:
                conv_signal = conv_signal[:, :, :-(self._pad//self._stride)]
        return conv_signal


class ComplexConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            pad_mode="causal",
            dilation=1,
            groups=1,
            complex_axis=0,
    ):
        '''
            Complex 1D CNN. If the complex_axix=0 (batch axis), the first half dimension is real part,
            the second half is imaginary part.
            in_channels: real+imag
            out_channels: real+imag
            kernel_size:
            padding:
        '''
        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode

        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = ConvNorm(self.in_channels, self.out_channels, kernel_size, self.stride,
                                  pad_mode=pad_mode, dilation=self.dilation, groups=self.groups)
        self.imag_conv = ConvNorm(self.in_channels, self.out_channels, kernel_size, self.stride,
                                  pad_mode=pad_mode, dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
            real = real2real - imag2imag
            imag = real2imag + imag2real
            out = torch.cat([real, imag], self.complex_axis)
        else:
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
            real2real = self.real_conv(real[:, 0])
            imag2imag = self.imag_conv(imag[:, 0])

            real2imag = self.imag_conv(real[:, 0])
            imag2real = self.real_conv(imag[:, 0])
            real = real2real - imag2imag
            imag = real2imag + imag2real
            real = real.unsqueeze(self.complex_axis)
            imag = imag.unsqueeze(self.complex_axis)
            out = torch.cat([real, imag], self.complex_axis)
        return out


class ComplexConvTranspose1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=8,
            stride=2,
            padding=3,
            complex_axis=0,
            groups=1
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose1d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose1d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, groups=self.groups)
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55

class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=0):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):
        # self._check_input_dim(xr, xi)
        xr, xi = torch.chunk(inputs, 2, dim=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze().detach().float(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze().detach().float(), exponential_average_factor)
                # self.RMr = self.RMr + exponential_average_factor * Mr.squeeze()
                # self.RMi = self.RMi + exponential_average_factor * Mi.squeeze()
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze().detach().float(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze().detach().float(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze().detach().float(), exponential_average_factor)
                # self.RVrr = self.RVrr + exponential_average_factor * Vrr.squeeze()
                # self.RVri = self.RVri + exponential_average_factor * Vri.squeeze()
                # self.RVii = self.RVii + exponential_average_factor * Vii.squeeze()
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


class WNCell(nn.Module):
    """ WaveNet-like cell """
    def __init__(self, residual_dim, gate_dim, skip_dim=128, cond_dim=0,
                 kernel_size=3, dilation=1, pad_mode='same'):
        """ Initialize WNCell module
        Args:
            residual_dim (int): Number of channels for residual connection.
            gate_dim (int): Number of channels for gate connection.
            skip_dim (int): Number of channels for skip connection.
            cond_dim (int): Number of channels for conditioning variables.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            pad_mode (str): Padding mode:
                "same": input and output frame length is same
                "causal": output only depends on current and previous input frames

        """
        super(WNCell, self).__init__()
        self.hidden_dim = gate_dim
        self.cond_dim = cond_dim
        self.dilation = dilation

        self.in_layer = nn.Sequential(
            ConvNorm(residual_dim,
                     2 * gate_dim,
                     kernel_size=kernel_size,
                     dilation=dilation,
                     pad_mode=pad_mode),
            nn.BatchNorm1d(2 * gate_dim, momentum=0.25),
        )
        if cond_dim > 0:
            self.conv_fuse = ConvNorm(2 * gate_dim, 2 * gate_dim, kernel_size=1,
                                      groups=2)

        self.res_layer = nn.Sequential(
            ConvNorm(gate_dim, residual_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(residual_dim, momentum=0.25)
        )

        self.skip_layer = nn.Sequential(
            ConvNorm(gate_dim, skip_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(skip_dim, momentum=0.25)
        )

    def forward(self, x, cond=None):
        """ Calculate forward propagation

        Args:
             x (Tensor): input variable
             cond (Tensor): condition variable

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        if self.cond_dim > 0:
            assert cond is not None
            acts = self.conv_fuse(self.in_layer(x) + cond)
        else:
            acts = self.in_layer(x)

        tanh_act = torch.tanh(acts[:, :self.hidden_dim, :])
        sigmoid_act = torch.sigmoid(acts[:, self.hidden_dim:, :])
        acts = tanh_act * sigmoid_act
        skip = self.skip_layer(acts)
        res = self.res_layer(acts)
        return (x + res) * math.sqrt(0.5), skip


class ComplexWNCell(nn.Module):
    """ WaveNet-like cell """
    def __init__(self, residual_dim, gate_dim, skip_dim=128,
                 kernel_size=3, dilation=1, pad_mode='same', complex_axis=0):
        """ Initialize ComplexWNCell module
        Args:
            residual_dim (int): Number of channels for residual connection.
            gate_dim (int): Number of channels for gate connection.
            skip_dim (int): Number of channels for skip connection.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            pad_mode (str): Padding mode:
                "same": input and output frame length is same
                "causal": output only depends on current and previous input frames

        """
        super(ComplexWNCell, self).__init__()
        self.hidden_dim = gate_dim
        self.dilation = dilation

        self.in_layer = nn.Sequential(
            ComplexConv1d(residual_dim,
                          2 * gate_dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          pad_mode=pad_mode,
                          complex_axis=complex_axis),
            ComplexBatchNorm(2 * gate_dim, momentum=0.25),
        )
        self.res_layer = nn.Sequential(
            ComplexConv1d(gate_dim, residual_dim,
                          kernel_size=kernel_size, pad_mode=pad_mode,
                          complex_axis=complex_axis),
            ComplexBatchNorm(residual_dim, momentum=0.25)
        )

        self.skip_layer = nn.Sequential(
            ComplexConv1d(gate_dim, skip_dim, kernel_size=kernel_size,
                          pad_mode=pad_mode, complex_axis=complex_axis),
            ComplexBatchNorm(skip_dim, momentum=0.25)
        )

    def forward(self, x, cond=None):
        """ Calculate forward propagation

        Args:
             x (Tensor): input variable
             cond (Tensor): condition variable

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        acts = self.in_layer(x)
        tanh_act = torch.tanh(acts[:, :self.hidden_dim, :])
        sigmoid_act = torch.sigmoid(acts[:, self.hidden_dim:, :])
        acts = tanh_act * sigmoid_act
        skip = self.skip_layer(acts)
        res = self.res_layer(acts)

        return (x + res) * math.sqrt(0.5), skip


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also q latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized


class VectorQuantize(nn.Module):
    """ Vector Quantization modules with straight-through trick """
    def __init__(self, emb_dim, n_emb):
        """ Initialize Vector Quantization module
        Args:
             emb_dim (int): Number of channels of embedding vector
             n_emb (int): Number of embedding in codebook

        """
        super(VectorQuantize, self).__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.codebook = Parameter(torch.Tensor(emb_dim, n_emb).uniform_(-1/n_emb, 1/n_emb))
        self.emb_dim = emb_dim
        self.jitter = Jitter()

    def forward(self, z_e_x, jitter=False):
        """ Calculate forward propagation
        Args:
            z_e_x (Tensor): input tensor for quantization
            jitter (Bool): Set to True for using jitter
        """
        inputs_size = z_e_x.size()
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous().view(-1, self.emb_dim)
        dist2 = torch.sum(z_e_x_ ** 2, 1, keepdim=True) \
            - 2 * torch.matmul(z_e_x_, self.codebook) \
            + torch.sum(self.codebook ** 2, 0)
        _, z_id_flatten = torch.max(-dist2, dim=1)
        z_id = z_id_flatten.view(inputs_size[0], inputs_size[2])
        z_q_flatten = torch.index_select(self.codebook.t(), dim=0, index=z_id_flatten)
        z_q = z_q_flatten.view(inputs_size[0], inputs_size[2], self.emb_dim).permute(0, 2, 1)
        if jitter:
            z_q = self.jitter(z_q)
        return z_q, z_id


class Encoder(nn.Module):
    """ Encoder module with skip and residual connection """
    def __init__(self, input_dim, output_dim,
                 residual_dim, gate_dim, skip_dim,
                 kernel_size, n_stage=1, cond_dim=0,
                 down_sample_factor=2,
                 pad_mode='same',
                 dilation_rate=None):
        """ Initialize Encoder module

        Args:
            input_dim (int): Number of channels of input tensor
            output_dim (int): Number of channels of output tensor
            skip_dim (int): Number of channels of skip connection
            kernel_size (int): Size of kernel
            down_sample_factor: Upsample factor
            dilation_rate: List of dilation rate for WNCel
            pad_mode: padding mode
                "same": same padding
                "causal": causal padding

        Returns:
            Tensor: Output tensor

        """
        super().__init__()
        self.gate_dim = gate_dim
        self.residual_dim = residual_dim
        self.skip_dim = skip_dim
        self.down_sample_factor = down_sample_factor
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 8, 16]
        self.input_layer = nn.Sequential(
            ConvNorm(input_dim, residual_dim, kernel_size=5,
                     pad_mode=pad_mode),
            nn.BatchNorm1d(residual_dim, momentum=0.8),
            nn.PReLU(residual_dim)
        )
        if self.down_sample_factor > 1:
            self.down_sample = nn.ModuleList()
            assert down_sample_factor % 2 == 0
            for i in range(down_sample_factor//2):
                self.down_sample.extend([ConvNorm(residual_dim, residual_dim,
                                                  kernel_size=8, stride=2,
                                                  pad_mode=pad_mode),
                                         nn.BatchNorm1d(residual_dim, momentum=0.8),
                                         nn.PReLU(residual_dim)
                                         ])
            self.down_sample = nn.Sequential(*self.down_sample)

        self.WN = nn.ModuleList()
        for i in range(n_stage):
            for d in dilation_rate:
                self.WN.append(WNCell(residual_dim=residual_dim,
                                      gate_dim=gate_dim,
                                      skip_dim=skip_dim,
                                      kernel_size=kernel_size,
                                      cond_dim=cond_dim,
                                      dilation=d,
                                      pad_mode=pad_mode))

        self.output_layer = nn.Sequential(
            ConvNorm(skip_dim, output_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(output_dim, momentum=0.8),
            nn.PReLU(output_dim),
            ConvNorm(output_dim, output_dim, kernel_size=1)
        )

        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_layer = nn.Sequential(
                ConvNorm(cond_dim,
                         2 * gate_dim * len(self.WN),
                         kernel_size=3,
                         pad_mode=pad_mode),
                nn.BatchNorm1d(2 * gate_dim * len(self.WN),
                               momentum=0.25),
                nn.ReLU()
            )

    def forward(self, x, cond=None):
        """ Calculate forward propagation
        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor

        """
        h = self.input_layer(x)
        if self.cond_dim > 0:
            assert cond is not None
            h_cond = self.cond_layer(cond)

        if self.down_sample_factor > 1:
            h = self.down_sample(h)

        skip = 0

        if self.cond_dim > 0:
            for i in range(len(self.WN)):
                h, _skip = self.WN[i](h, h_cond[:, 2*i*self.gate_dim: 2*(i+1)*self.gate_dim])
                skip += _skip
        else:
            for i in range(len(self.WN)):
                h, _skip = self.WN[i](h)
                skip += _skip
        skip *= math.sqrt(1.0 / len(self.WN))
        output = self.output_layer(skip)
        return output


class Decoder(nn.Module):
    """ Decoder module with residual-skip connection """
    def __init__(self, input_dim,
                 output_dim,
                 residual_dim,
                 gate_dim,
                 skip_dim,
                 n_stage,
                 kernel_size,
                 cond_dim=0,
                 n_upsample_factor=2,
                 pad_mode='same',
                 dilation_rate=None):
        """ Initialize Decoder module

        Args:
            input_dim (int): Number of channels of input tensor
            output_dim (int): Number of channels of output tensor
            skip_dim (int): Number of channels of skip connection
            n_stage (int): Number of dilated WNCell stage
            kernel_size (int): Size of kernel
            n_upsample_factor: Upsample factor
            dilation_rate: List of dilation rate for WNCell

        Returns:
            Tensor: Output tensor

        """
        super(Decoder, self).__init__()
        self.residual_dim = residual_dim
        self.gate_dim = gate_dim
        self.pad_mode = pad_mode
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 8, 16, 32]
        # assert n_upsample_factor % 2 == 0
        self.input_layer = nn.Sequential(
            ConvNorm(in_channels=input_dim, out_channels=2*residual_dim,
                     kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(2*residual_dim, momentum=0.25),
            nn.GLU(dim=1)
        )

        self.upsample = nn.ModuleList()
        if pad_mode == 'causal':
            upsample_padding = 0
        else:
            upsample_padding = 3
        for i in range(n_upsample_factor // 2):
            self.upsample.append(nn.Sequential(nn.ConvTranspose1d(residual_dim,
                                                                  2*residual_dim,
                                                                  kernel_size=8,
                                                                  stride=2,
                                                                  padding=upsample_padding),
                                               nn.BatchNorm1d(2*residual_dim, momentum=0.25),
                                               nn.GLU(dim=1)))

        self.WN = nn.ModuleList()
        for i in range(n_stage):
            for d in dilation_rate:
                self.WN.append(WNCell(residual_dim=residual_dim,
                                      gate_dim=gate_dim,
                                      skip_dim=skip_dim,
                                      cond_dim=cond_dim,
                                      kernel_size=kernel_size,
                                      dilation=d, pad_mode=pad_mode))

        self.output_layer = nn.Sequential(
            ConvNorm(skip_dim, 2 * output_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(2 * output_dim, momentum=0.25),
            nn.GLU(dim=1),
            ConvNorm(output_dim, output_dim, kernel_size=15, pad_mode=pad_mode),
        )
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_layer = nn.Sequential(
                ConvNorm(cond_dim,
                         2 * gate_dim * len(self.WN),
                         kernel_size=3,
                         pad_mode=pad_mode),
                nn.BatchNorm1d(2 * gate_dim * len(self.WN), momentum=0.25),
                nn.PReLU()
            )

    def forward(self, x, cond=None):
        """ Calculate forward pass

        Args:
            x (Tensor): input tensor
            cond_in (Tensor): condition tensor

        Returns:
            Tensor: Output tensor
        """
        h = self.input_layer(x)
        if self.cond_dim > 0:
            assert cond is not None
            h_cond = self.cond_layer(cond)

        for upsample in self.upsample:
            h = upsample(h)
            if self.pad_mode == 'causal':
                h = h[:, :, :2*x.shape[2]]
        skip = 0
        if self.cond_dim > 0:
            for i in range(len(self.WN)):
                h, _skip = self.WN[i](h, h_cond[:, 2 * i * self.gate_dim: 2*(i + 1)*self.gate_dim])
                skip += _skip
        else:
            for i in range(len(self.WN)):
                h, _skip = self.WN[i](h)
                skip += _skip
        # Normalizing value
        skip *= math.sqrt(1.0 / len(self.WN))
        output = self.output_layer(skip)

        return output


class PreEmphasis(torch.nn.Module):
    """ Adapt from https://github.com/clovaai/voxceleb_trainer/blob/master/utils.py """
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.size()) == 2, 'The number of dimensions of input tensor must be 2! Input shape %d' % x.shape[-1]
        # reflect padding to match lengths of in/out
        x = x.unsqueeze(1)
        x = F.pad(x, [1, 0], 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)

