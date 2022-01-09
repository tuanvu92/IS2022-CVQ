import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import pdb


def adaptive_s_norm(emb, imposter_cohort, imposter_size=100):
    # emb.shape = [B, 192]
    # imposter_cohort.shape = [N, 192]
    emb = F.normalize(emb, p=2, dim=-1)
    imposter_cohort = F.normalize(imposter_cohort, p=2, dim=-1)

    similarity = emb @ imposter_cohort.T
    # Select imposter with highest similarity
    # similarity.shape = [B, N] -> [B, imposter_size]
    similarity, _ = similarity.topk(imposter_size, dim=-1)
    mean = similarity.mean(dim=-1, keepdim=True)
    std = similarity.std(dim=-1, keepdim=True)

    score_t_norm = (emb @ emb.T - mean) / std
    score_s_norm = 0.5 * (score_t_norm + score_t_norm.T)
    return score_s_norm


def calculate_err(emb, file_list, pseudo_emb):
    speaker_emb_list = dict()
    for fname, _emb in zip(file_list, emb):
        speaker = fname.split("/")[-2]
        if speaker not in speaker_emb_list:
            speaker_emb_list[speaker] = [_emb]
        else:
            speaker_emb_list[speaker].append(_emb)
    emb_all = []
    speaker_id_range = dict()
    idx = 0

    for speaker in speaker_emb_list:
        emb_all.extend(speaker_emb_list[speaker])
        speaker_id_range[speaker] = [idx, idx + len(speaker_emb_list[speaker])]
        idx += len(speaker_emb_list[speaker])

    emb_all = torch.stack(emb_all, dim=0)
    s_matrix = adaptive_s_norm(emb_all, imposter_cohort=pseudo_emb)
    s_matrix = s_matrix.numpy()
    # s_matrix to list
    # score = []
    # ground_truth = []
    # for speaker, (start, end) in speaker_id_range.items():
    #     s_flatten = s_matrix[start:end, start:end].flatten()
    #     score.extend(s_flatten)
    #     ground_truth.extend(np.ones_like(s_flatten))
    #     if start > 0:
    #         s_flatten = s_matrix[0:start, :].flatten()
    #         score.extend(s_flatten)
    #         ground_truth.extend(np.zeros_like(s_flatten))
    # fpr, tpr, thresholds = roc_curve(ground_truth, score, pos_label=1)
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    eer, thresh, far, frr = search_equilibrium_point(np.min(s_matrix),
                                                 np.max(s_matrix * (1-np.eye(s_matrix.shape[0]))),
                                                 100, s_matrix, speaker_id_range)
    return eer, thresh, s_matrix


def search_equilibrium_point(low, high, n_step,
                             s_matrix, speaker_id_range):
    assert high > low
    diff = 1
    EER = 0
    EER_thres = 0
    M = s_matrix.shape[0]
    # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
    for thres in np.linspace(low, high, num=n_step):
        s_thres = s_matrix > thres
        # FPR = FP / (FP + TN)
        FP = 0
        FN = 0
        N = 0
        P = 0
        for speaker in speaker_id_range.keys():
            start, end = speaker_id_range[speaker]
            p = (end - start) ** 2
            n = ((end - start) * M - p) * 2
            tp = np.sum(s_thres[start:end, start:end])
            fn = p - tp
            assert fn >= 0
            fp = (np.sum(s_thres[start:end, :]) - tp) * 2
            assert fp >= 0
            FP += fp
            FN += fn
            P += p
            N += n
        FAR = FP / N
        FRR = FN / P
        # Save threshold when FAR = FRR (=EER)
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = (FAR + FRR) / 2
            EER_thres = thres
    return EER, EER_thres, FAR, FRR