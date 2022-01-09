import torch
import numpy as np
from scipy.signal import get_window
import librosa
import librosa.util as librosa_util
from gammatone.gtgram import gtgram
import scipy


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def mel2mcc(mel):
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    N = mel.shape[1]
    mel = mel.transpose(1, 2).unsqueeze(-1)
    mel = torch.cat([mel, torch.zeros_like(mel)], dim=-1)
    mcc = torch.irfft(mel, signal_ndim=1, signal_sizes=[2 * (N - 1)]).transpose(1, 2)[:, :N]
    mcc[:, 0] /= 2.
    return mcc.squeeze()


def mel2mcc_np(mel):
    if len(mel.shape) == 2:
        mel = np.expand_dims(mel, 0)
    c = np.fft.irfft(mel, axis=1)
    c[:, 0] /= 2.0
    c = c[:, :mel.shape[1]]
    return np.squeeze(c)


def mcc2mel(mcc):
    if len(mcc.shape) == 2:
        mcc = mcc.unsqueeze(0)
    mcc[:, 0] *= 2.
    mcc = mcc.transpose(1, 2)
    mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
    mel = torch.rfft(mcc, signal_ndim=1)[:, :, :, 0]
    return mel.transpose(1, 2).squeeze()


def np_mel2mcc(mel):
    c = np.fft.irfft(mel, axis=0)
    c[0] /= 2.0
    return c[:mel.shape[0]]


def np_mcc2mel(mcc):
    sym_c = np.zeros([2*(mcc.shape[0]-1), mcc.shape[1]])
    sym_c[0] = 2*mcc[0]
    for i in range(1, mcc.shape[0]):
        sym_c[i] = mcc[i]
        sym_c[-i] = mcc[i]

    mel = np.fft.rfft(sym_c, axis=0).real
    return mel


def gfcc(x, fs, n_channels=80, f_min=80, win_length=0.025, hop_length=0.01):
    s = gtgram(x, fs, win_length, hop_length, n_channels, f_min)
    gfc = np.fft.irfft(np.log(s), axis=0)
    gfc[0] /= 2.0
    gfc = gfc[:s.shape[0]]
    return gfc


def fft2gammatonemx(nfft: int, sr: int, nfilts: int = 64, width: float = 1.0,
                    minfreq: float = 100., maxfreq: float = None):
    if maxfreq is None:
        maxfreq = sr / 2.

    j = complex(0, 1)

    wts = np.zeros((nfilts, nfft // 2 + 1))

    EarQ = 9.26449
    minBW = 24.7
    order = 1

    cfreqs = -(EarQ * minBW) + np.exp(
        np.arange(1, nfilts + 1) * (-np.log(maxfreq + EarQ * minBW) + np.log(minfreq + EarQ * minBW)) / nfilts
    ) * (maxfreq + EarQ * minBW)
    cfreqs = cfreqs[::-1]
    GTord = 4
    ucirc = np.exp(j * 2. * np.pi * np.arange(0, nfft // 2 + 1) / nfft)
    for i in range(nfilts):
        cf = cfreqs[i]
        ERB = width * ((cf / EarQ) ** order + minBW ** order) ** (1. / order)
        B = 1.019 * 2. * np.pi * ERB
        r = np.exp(-B / sr)
        theta = 2. * np.pi * cf / sr
        pole = r * np.exp(j * theta)

        T = 1. / sr
        A11 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
                + 2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
        A12 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
                - 2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
        A13 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
                + 2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
        A14 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
                - 2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
        zros = -np.array([A11, A12, A13, A14]) / T
        gain = abs((
                           - 2 * np.exp(4 * j * cf * np.pi * T) * T
                           + 2 * np.exp(-(B * T) + 2 * j * cf * np.pi * T) * T * (
                                   np.cos(2 * cf * np.pi * T)
                                   - np.sqrt(3 - 2 ** (3. / 2)) * np.sin(2 * cf * np.pi * T)
                           )
                   ) * (
                           - 2 * np.exp(4 * j * cf * np.pi * T) * T
                           + 2 * np.exp(-(B * T) + 2 * j * cf * np.pi * T) * T * (
                                   np.cos(2 * cf * np.pi * T)
                                   + np.sqrt(3 - 2 ** (3. / 2)) * np.sin(2 * cf * np.pi * T)
                           )
                   ) * (
                           - 2 * np.exp(4 * j * cf * np.pi * T) * T
                           + 2 * np.exp(-(B * T) + 2 * j * cf * np.pi * T) * T * (
                                   np.cos(2 * cf * np.pi * T)
                                   - np.sqrt(3 + 2 ** (3. / 2)) * np.sin(2 * cf * np.pi * T)
                           )
                   ) * (
                           - 2 * np.exp(4 * j * cf * np.pi * T) * T
                           + 2 * np.exp(-(B * T) + 2 * j * cf * np.pi * T) * T * (
                                   np.cos(2 * cf * np.pi * T)
                                   + np.sqrt(3 + 2 ** (3. / 2)) * np.sin(2 * cf * np.pi * T)
                           )
                   ) / (
                           - 2. / np.exp(2 * B * T)
                           - 2 * np.exp(4 * j * cf * np.pi * T)
                           + 2 * (1 + np.exp(4 * j * cf * np.pi * T)) / np.exp(B * T)
                   ) ** 4
                   )
        wts[i, :] = (
                ((T ** 4) / gain) * np.abs(ucirc - zros[0]) * np.abs(ucirc - zros[1]) * np.abs(ucirc - zros[2])
                * np.abs(ucirc - zros[3]) * (np.abs((pole - ucirc) * (np.conjugate(pole) - ucirc)) ** (-GTord))
        )
    return wts, cfreqs
