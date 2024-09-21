# Standard Library Imports
import argparse
import copy
import glob
import json
import logging
import os
import os.path as osp
import random
import sys
import time
from typing import Union

# Third-Party Imports
import librosa
import librosa.util as librosa_util
import munch
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import yaml
from attrdict import AttrDict
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.io.wavfile import write
from scipy.signal import get_window
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import (
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    AvgPool1d,
)
from torch.nn.utils import (
    weight_norm,
    remove_weight_norm,
    spectral_norm,
)
from torch.utils.data import DataLoader
from transformers import (
    AlbertConfig,
    AlbertModel,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)
from torchaudio.models import Conformer

# Local Application/Specific Imports
from monotonic_align import maximum_path, mask_from_lens
from monotonic_align.core import maximum_path_c
# from meldataset_PH_new import build_dataloader
from StyleTTS.models import *
from StyleTTS.optimizers import build_optimizer
from StyleTTS.Utils.ASR.models import ASRCNN
from StyleTTS.Utils.JDC.model import JDCNet
from StyleTTS.Demo.hifi_gan.vocoder import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from audio_diffusion_pytorch.modules import *
from StyleTTS.Demo.hifi_gan.vocoder_utils import get_padding, init_weights

# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Additional Setup
sys.path.insert(0, "/share/ctn/users/yl4579/.GAN/hifi-gan_old")

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        # train_path = "Data_LibriTTS_R_comb_ref/train_list.txt"
        train_path = "StyleTTS/Data/train_list.txt"
    if val_path is None:
        # val_path = "Data_LibriTTS_R_comb_ref/val_list.txt"
        val_path = "StyleTTS/Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    # train_list = train_list[-1000:]
    # val_list = train_list[:1000]
    return train_list, val_list




class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes



def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l[:-1].split('|') for l in data_list]
        # self.data_list = [data if len(data) == 5 else (*data, 0) for data in _data_list]
        self.data_list = [
                data if len(data) == 5 else (
                    [data[0], "", data[1], data[2], ""] if len(data) == 3 else data + ["", ""])
                for data in _data_list
            ]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
#         self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        try:
            wave, text_tensor, speaker_id = self._load_tensor(data)
        except:
            print(path)
            return self.__getitem__(idx - 1 if idx > 1 else len(self.data_list) - 1)
        if 'LibriTTS_R' in path:
            wave = wave[..., :-50]
        
        mel_tensor = preprocess(wave).squeeze()
        
        try:
#             if bool(random.getrandbits(1)):
            adj_path = data[1]
            adj_txt = data[-1]
            
            if adj_txt == "" or "|" in adj_txt or "    " in adj_text:
                mel_adj = mel_tensor
                text_adj = text_tensor
            else:
            
                adj_wave, sr = sf.read(adj_path)
                if adj_wave.shape[-1] == 2:
                    adj_wave = wave[:, 0].squeeze()
                if sr != 24000:
                    adj_wave = librosa.resample(adj_wave, sr, 24000)
                    print(adj_path, sr)

                adj_wave = np.concatenate([np.zeros([5000]), adj_wave, np.zeros([5000])], axis=0)

                mel_adj = preprocess(adj_wave).squeeze()
                
                text = self.text_cleaner(adj_txt)

                text.insert(0, 0)
                text.append(0)

                text_adj = torch.LongTensor(text)

        except:
            mel_adj = mel_tensor
            text_adj = text_tensor
#         if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
#             mel_tensor = F.interpolate(
#                 mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
#                 mode='linear').squeeze(0)
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        acoustic_adj = mel_adj.squeeze()
        length_adj = acoustic_adj.size(1)
        acoustic_adj = acoustic_adj[:, :(length_adj - length_adj % 2)]
        
        try:
            ref_data = random.choice(self.data_list)
            ref_mel_tensor, ref_label = self._load_data(ref_data)
        except:
            print(ref_data[0])
            return self.__getitem__(idx - 1 if idx > 1 else len(self.data_list) - 1)
        
        return speaker_id, acoustic_feature, text_tensor, ref_mel_tensor, ref_label, path, wave, acoustic_adj, text_adj

    def _load_tensor(self, data):
        wave_path, _, text, speaker_id, _ = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, sr, 24000)
            print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rt_length = max([b[-1].shape[0] for b in batch])

        max_ref_length = max([b[-2].shape[1] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        
        ref_texts = torch.zeros((batch_size, max_rt_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        adj_mels = torch.zeros((batch_size, nmels, max_ref_length)).float()
        adj_mels_lengths = torch.zeros(batch_size).long()
        
        for bid, (label, mel, text, ref_mel, ref_label, path, wave, adj_mel, adj_text) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rt_size = adj_text.size(0)

            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rt_size] = adj_text

            input_lengths[bid] = text_size
            ref_lengths[bid] = rt_size
            
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            adj_mels_size = adj_mel.size(1)
            adj_mels[bid, :, :adj_mels_size] = adj_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave
            adj_mels_lengths[bid] = adj_mels_size
            
#             if(text_size < (mel_size//2)):
#                 print(text_size, mel_size//2)

        return waves, texts, input_lengths, mels, output_lengths, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader



        
class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),  # out: (b, 64, 31, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 31, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  # (b, 128, 31, 128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)  # (b, 192, 31, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 31, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.2),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        # in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        # in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        # in = (b, 640, 31, 2), out = (b, 256, 31, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.2),
        )

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, bidirectional=True)  # (b, 31, 512)

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, bidirectional=True)  # (b, 31, 512)

        # input: (b * 31, 512)
        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)  # (b * 31, num_class)

        # input: (b * 31, 512)
        self.detector = nn.Linear(in_features=512, out_features=2)  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)
        
    def forward(self, x, input_lengths):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 31, 722), (b, 31, 2)
        """
        ###############################
        # forward pass for classifier #
        ###############################
        seq_len = x.shape[-1]
        x = x.float().transpose(-1, -2)
        
        convblock_out = self.conv_block(x)
        
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        
        
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        GAN_feature = poolblock_out.transpose(-1, -2)
        poolblock_out = self.pool_block[2](poolblock_out)
        
        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        
        classifier_out = nn.utils.rnn.pack_padded_sequence(
            classifier_out, input_lengths, batch_first=True, enforce_sorted=False)
        
        self.bilstm_classifier.flatten_parameters()
        classifier_out, _ = self.bilstm_classifier(classifier_out)  # ignore the hidden states
        classifier_out, _ = nn.utils.rnn.pad_packed_sequence(
            classifier_out, batch_first=True)
        
        classifier_out = classifier_out.contiguous().view((-1, 512))  # (b * 31, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, seq_len, self.num_class))  # (b, 31, num_class)
        
        # sizes: (b, 31, 722), (b, 31, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return torch.abs(classifier_out.squeeze())

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)
                    

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x
    



def _load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    model_config = config['model_params']
    return model_config

def _load_model(model_config, model_path):
    model = ASRCNN(**model_config)
    params = torch.load(model_path, map_location='cpu')['model']
    model.load_state_dict(params)
    return model



def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x






def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
            return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.abs(x_stft).transpose(2, 1)

class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
            ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    


class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)



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


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.to(inverse_transform.device()) if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                                 device=f0.device)
            # fundamental component
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

            # generate sine waveforms
            sine_waves = self._f02sine(fn) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)


class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    harmonic_num=8, voiced_threshod=10)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel//(2**i), 
                         upsample_initial_channel//(2**(i+1)),
                         k, u, padding=(u//2 + u%2), output_padding=u%2)))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
                self.noise_res.append(resblock(c_cur, 11, [1,3,5], style_dim))
            
        self.resblocks = nn.ModuleList()
        
        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, style_dim))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, s, f0):
        
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        
        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)
            
            x = self.ups[i](x)
            x = x + x_source
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = x + (1 / self.alphas[i+1]) * (torch.sin(self.alphas[i+1] * x) ** 2)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)



class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Decoder(nn.Module):
    def __init__(self, dim_in=512, F0_channel=512, style_dim=64, dim_out=80, 
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [10,5,3,2],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[20,10,6,4]):
        super().__init__()
        
        self.decode = nn.ModuleList()
        
        self.conformer = Conformer(
             input_dim=dim_in + 2,
             num_heads=2,
             ffn_dim=dim_in * 2,
             num_layers=1,
             depthwise_conv_kernel_size=7,
             use_group_norm=True,
        )
        
        self.encode = AdainResBlk1d(dim_in + 4, 1024, style_dim)
        
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )
        
        
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes)

        
    def forward(self, asr, F0_curve, N, s):
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = nn.functional.conv1d(F0_curve.unsqueeze(1), torch.ones(1, 1, F0_down).to('cuda'), padding=F0_down//2).squeeze(1) / F0_down
            if N_down:
                N = nn.functional.conv1d(N.unsqueeze(1), torch.ones(1, 1, N_down).to('cuda'), padding=N_down//2).squeeze(1)  / N_down

        
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        
        x = torch.cat([asr, F0, N], axis=1)
        input_lengths = torch.ones(x.size(0)) * x.size(-1)
        x, _ = self.conformer(x.transpose(-1, -2), input_lengths.to(x.device))
        x = torch.cat([x.transpose(-1, -2), F0, N], axis=1)
        x = self.encode(x, s)
        
        asr_res = self.asr_res(asr)
        
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
                
        x = self.generator(x, s, F0_curve)
        return x
    
    def feature(self, asr, F0_curve, N, input_lengths):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        
        x = torch.cat([asr, F0, N], axis=1)
        x, _ = self.conformer(x.transpose(-1, -2), input_lengths)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask



class AttentionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        return x
    

class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: Optional[int] = None,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features
            
        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        # Compute and return attention
        return self.attention(q, k, v)

    
class CrossAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        num_time: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_key = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_k = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, features)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        idx = torch.arange(0, self.num_time).to(x.device)
        keys = self.positions(idx).transpose(-1, -2).expand(x.shape[0], -1, -1).transpose(-1, -2)
        
        # Normalize then compute q from input and k,v from context
        x, keys, context = self.norm(x), self.norm(keys), self.norm_context(context)
        q, k, v = (self.to_q(x), self.to_k(keys), self.to_v(context))

        # Compute and return attention
        return self.attention(q, k, v)



class TVStyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, 
                 num_heads=8, num_time=50, num_layers=6,
                 head_features=64):
        super().__init__()
        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        
        max_conv_dim = text_dim
        
        self.cross_attention = Attention(
            features=max_conv_dim,
            num_heads=num_heads,
            head_features=head_features,
            context_features=max_conv_dim,
            use_rel_pos=False
        )
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, max_conv_dim)
        
        self.embedder = NumberEmbedder(features=max_conv_dim)
        
    def forward(self, x, input_lengths):
        x = x[..., :input_lengths.max()]
        
        x = self.mel_proj(x)
        x = x.transpose(-1, -2)
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        h = x.transpose(-1, -2)
        
        idx = torch.arange(0, self.num_time).to(x.device)
        positions = self.positions(idx).transpose(-1, -2).expand(x.shape[0], -1, -1)
        
        pos = self.embedder(torch.arange(h.shape[-1]).to(x.device)).transpose(-1, -2).expand(h.size(0), -1, -1)
        h += pos
                
        m = length_to_mask(input_lengths).to(x.device)
        h.masked_fill_(m.unsqueeze(1), 0.0)
        h = self.cross_attention(positions.transpose(-1, -2), context=h.transpose(-1, -2))
        
        return h.transpose(-1, -2)
    
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
    
class DurationPredictor(nn.Module):
    def __init__(self, num_heads=8, head_features=64, d_hid=512, nlayers=6, max_dur=50):
        super().__init__()
        
        self.transformer = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers,
             depthwise_conv_kernel_size=7,
             use_group_norm=True,
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
    def forward(self, text, style, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel_len = style.size(-1) # length of mel
        input_lengths = input_lengths + mel_len
        x = torch.cat([style, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, _ = self.transformer(x, input_lengths)
        x = x.transpose(-1, -2)
        x_text = x[:, :, mel_len:]
        text_return[:, :, :x_text.size(-1)] = x_text        
        out = self.duration_proj(text_return.transpose(-1, -2))
        return out
    
class ProsodyPredictor(nn.Module):
    def __init__(self, num_heads=8, head_features=64, d_hid=512, nlayers=6, scale_factor=2):
        super().__init__()
        
        self.conf_pre = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers // 2,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        )
                
        self.conf_after = Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=nlayers // 2,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,
        )
        
        self.F0_proj = LinearNorm(d_hid, 1)
        self.N_proj = LinearNorm(d_hid, 1)
        
        self.scale_factor = scale_factor
        
    def forward(self, text, style, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size * self.scale_factor).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel_len = style.size(-1) # length of mel
        input_lengths = input_lengths + mel_len
        x = torch.cat([style, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, _ = self.conf_pre(x, input_lengths)
        x = F.interpolate(x.transpose(-1, -2), scale_factor=self.scale_factor, mode='nearest').transpose(-1, -2)
        x, _ = self.conf_after(x, input_lengths * self.scale_factor)
        
        x = x.transpose(-1, -2)
        x_text = x[:, :, mel_len * self.scale_factor:]
        text_return[:, :, :x_text.size(-1)] = x_text  
        F0 = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        N = self.N_proj(text_return.transpose(-1, -2)).squeeze(-1)
        return F0, N
    




class MultiTaskModel(nn.Module):
    def __init__(self, model, dropout=0.1, num_tokens=178, num_vocab=593, hidden_size=768):
        super().__init__()

        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, word_embedding.shape[-1])
    
    def forward(self, phonemes):
        output = self.encoder(phonemes)
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)
        
        return tokens_pred, words_pred
    


class StyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Linear(text_dim, style_dim)
        
    def forward(self, mel, text, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel = self.mel_proj(mel)
        mel_len = mel.size(-1) # length of mel

        input_lengths = input_lengths + mel_len
        x = torch.cat([mel, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        x = x.transpose(-1, -2)
        x_mel = x[:, :, :mel_len]
        x_text = x[:, :, mel_len:]
        
        s = self.out(x_mel.mean(axis=-1))
        text_return[:, :, :x_text.size(-1)] = x_text
        
        
        return s, text_return
    



def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class VectorQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """

    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[2]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        """
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        encodings = rearrange(z_e, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=z.size(0))

        # vector quantization cost that trains the embedding vectors
        z_q = self.codebook(indices).transpose(1, 2)  # (B x D x T)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices

    def embed_code(self, embed_id):
        emb = F.embedding(embed_id, self.codebook.weight)
        return self.out_proj(emb.transpose(1, 2))
    
class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_codebooks: int = 7,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim)
                for i in range(n_codebooks)
            ]
        )

    def forward(self, z, n_quantizers: Union[None, int, Tensor] = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        for i, quantizer in enumerate(self.quantizers):
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.size(0),), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)

        return z_q, commitment_loss, codebook_loss, torch.stack(codebook_indices, dim=1)

    def encode(self, z, n_quantizers: int = None):
        residual = z
        codes = []
        for quantizer in self.quantizers[:n_quantizers]:
            z_q_i, _, _, indices_i = quantizer(residual)
            residual = residual - z_q_i

            codes.append(indices_i)

        return torch.stack(codes, dim=-1)

    def decode(self, codes):
        z_q = 0

        for i, indices_i in enumerate(codes.unbind(dim=-1)):
            z_q += self.quantizers[i].embed_code(indices_i)

        return z_q

## ?????
# class ProsodyDiscriminator(nn.Module):
#     def __init__(self, mel_dim=514, num_heads=8, head_features=64, d_hid=512, nlayers=6, scale_factor=1):
#         super().__init__()
        
#         self.mel_proj = nn.Conv1d(mel_dim, d_hid, kernel_size=3, padding=1)
#         self.d_hid = d_hid
        
#         self.conf_pre = torch.nn.ModuleList(
#             [Conformer(
#              input_dim=d_hid,
#              num_heads=num_heads,
#              ffn_dim=d_hid * 2,
#              num_layers=1,
#              depthwise_conv_kernel_size=15,
#              use_group_norm=True,)
#                 for _ in range(nlayers // 2)
#             ]
#         )
        
#         self.conf_after = torch.nn.ModuleList(
#             [Conformer(
#              input_dim=d_hid,
#              num_heads=num_heads,
#              ffn_dim=d_hid * 2,
#              num_layers=1,
#              depthwise_conv_kernel_size=15,
#              use_group_norm=True,)
#                 for _ in range(nlayers // 2)
#             ]
#         )
        
#         self.F0_proj = LinearNorm(d_hid, 1)
        
#         self.scale_factor = scale_factor
        
#     def forward(self, text, style, input_lengths, max_size):
#         text_return = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1)).to(text.device)
        
#         hidden = []
        
#         text = text[..., :input_lengths.max()]
#         text = self.mel_proj(text)
        
#         mel_len = style.size(-1) # length of mel
#         input_lengths = input_lengths + mel_len
#         x = torch.cat([style, text], axis=-1).transpose(-1, -2) # last dimension
        
#         for layer in (self.conf_pre):
#             x, _ = layer(x, input_lengths)
#             h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1)).to(text.device)
#             h[:, :, :x.size(-2)] = x.transpose(-1, -2)
#             hidden.append(h)
        
#         x = F.interpolate(x.transpose(-1, -2), scale_factor=self.scale_factor, mode='nearest').transpose(-1, -2)
#         for layer in (self.conf_after):
#             x, _ = layer(x, input_lengths * self.scale_factor)
#             h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1)).to(text.device)
#             h[:, :, :x.size(-2)] = x.transpose(-1, -2)  
#             hidden.append(h)
            
#         x = x.transpose(-1, -2)
#         text_return[:, :, :x.size(-1)] = x  
#         F0 = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        
#         return F0, hidden
    
class ProsodyDiscriminator(nn.Module):
    def __init__(self, mel_dim=514, num_heads=8, head_features=64, d_hid=512, nlayers=6, scale_factor=1):
        super().__init__()
        
        self.mel_proj = nn.Conv1d(mel_dim, d_hid, kernel_size=3, padding=1)
        self.d_hid = d_hid
        
        self.conf_pre = torch.nn.ModuleList(
            [Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=1,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,)
                for _ in range(nlayers // 2)
            ]
        )
        
        self.conf_after = torch.nn.ModuleList(
            [Conformer(
             input_dim=d_hid,
             num_heads=num_heads,
             ffn_dim=d_hid * 2,
             num_layers=1,
             depthwise_conv_kernel_size=15,
             use_group_norm=True,)
                for _ in range(nlayers // 2)
            ]
        )
        
        self.F0_proj = LinearNorm(d_hid, 1)
        self.sep = nn.Embedding(1, d_hid)
        
        self.scale_factor = scale_factor
        
    def forward(self, text, style, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
        
        hidden = []
        
        text = text[..., :input_lengths.max()]
        text = self.mel_proj(text)
        
        mel_len = style.size(-1) # length of mel
        input_lengths = input_lengths + mel_len + 1
        
        feat_sep = self.sep(torch.zeros(text.size(0)).long().to(text.device)).unsqueeze(-1)

        x = torch.cat([style, feat_sep, text], axis=-1).transpose(-1, -2) # last dimension
        
        for layer in (self.conf_pre):
            x, _ = layer(x, input_lengths)
            h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
            h[:, :, :x.size(-2)] = x.transpose(-1, -2)
            hidden.append(h)
        
        x = F.interpolate(x.transpose(-1, -2), scale_factor=self.scale_factor, mode='nearest').transpose(-1, -2)
        for layer in (self.conf_after):
            x, _ = layer(x, input_lengths * self.scale_factor)
            h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
            h[:, :, :x.size(-2)] = x.transpose(-1, -2)  
            hidden.append(h)
            
        x = x.transpose(-1, -2)
        text_return[:, :, :x.size(-1)] = x  
        F0 = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        
        return F0, hidden
    
def build_model():
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80).to('cuda')
    text_aligner = asr_model
    pitch_extractor = F0_model
    text_encoder = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).to('cuda')
    bert_encoder = nn.Linear(768, 512).to('cuda')

    style_encoder = StyleEncoder(mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6).to('cuda')  
    prosody_encoder = StyleEncoder(mel_dim=80, text_dim=512, style_dim=512, num_heads=8, num_layers=6).to('cuda')  

    dur_predictor = DurationPredictor(num_heads=8, head_features=64, d_hid=512, nlayers=6, max_dur=50).to('cuda')
    pro_predictor = ProsodyPredictor(num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    predictor_encoder = TVStyleEncoder(mel_dim=514, text_dim=512, 
                 num_heads=8, num_time=50, num_layers=6,
                 head_features=64).to('cuda')
    
    quantizer = ResidualVectorQuantize(
            input_dim=512, n_codebooks=9, codebook_dim=8
        ).to('cuda')
    
    text_embedding = nn.Embedding(512, 512).to('cuda')
    
    pro_discriminator = ProsodyDiscriminator(mel_dim=514, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    dur_discriminator = ProsodyDiscriminator(mel_dim=513, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    
    nets = Munch(bert=bert,
                 bert_encoder=bert_encoder,
                 text_embedding=text_embedding,
        decoder=decoder,
                 pitch_extractor=F0_model,
                     text_encoder=text_encoder,
                     style_encoder=style_encoder,
                 prosody_encoder=prosody_encoder,
predictor_encoder=predictor_encoder,
                 dur_predictor=dur_predictor,
                 pro_predictor=pro_predictor,
                quantizer=quantizer,
                discriminator=pro_discriminator,
                 dur_discriminator=dur_discriminator,
                 text_aligner = text_aligner,
                   mpd = MultiPeriodDiscriminator().to('cuda'),
                 msd = MultiResSpecDiscriminator().to('cuda')
                )

    return nets


_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

train_list, val_list = get_data_path_list()

batch_size = 16
device = 'cuda'

train_dataloader = build_dataloader(train_list,
                                    batch_size=batch_size,
                                    num_workers=8,
                                    dataset_config={},
                                    device=device)

val_dataloader = build_dataloader(val_list,
                                  batch_size=batch_size,
                                  validation=True,
                                  num_workers=2,
                                  device=device,
                                  dataset_config={})

i, (waves, texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths) = next(enumerate(val_dataloader))

# load F0 model

F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("StyleTTS/Utils/JDC/bst.t7")['net']
F0_model.load_state_dict(params)
_ = F0_model.train()
F0_model = F0_model.to('cuda')

F0_model_copy = copy.deepcopy(F0_model)

# load ASR model

ASR_MODEL_PATH = 'StyleTTS/Utils/ASR/epoch_00080.pth'
ASR_MODEL_CONFIG = 'StyleTTS/Utils/ASR/config.yml'

asr_model_config = _load_config(ASR_MODEL_CONFIG)
asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
_ = asr_model.train()
asr_model = asr_model.to('cuda')

asr_model_copy = copy.deepcopy(asr_model)

LRELU_SLOPE = 0.1

LRELU_SLOPE = 0.1

LRELU_SLOPE = 0.1

word_embedding = np.load('../../../SSL_Project/embedding_pca.npy')


albert_base_configuration = AlbertConfig(
    vocab_size=178,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=2048,
    max_position_embeddings=512,
    num_hidden_layers=12
)
model = AlbertModel(albert_base_configuration)
bert = MultiTaskModel(model).to('cuda')

checkpoint = torch.load("../../../SSL_Project/checkpoint_new/step_706000.t7", map_location='cpu')
state_dict = checkpoint['net']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
bert.load_state_dict(new_state_dict, strict=False)
print('Checkpoint loaded.')

bert = bert.encoder

model = build_model()


# params = torch.load("./checkpoint_LJ_TTS_PH_E2E_real_mel_snake_sigmoidur_R_newsine_conf_tvstyle_sup_codec/val_loss_tensor(2.2759, device='cuda:0').t7", map_location='cpu')
# params = torch.load("./checkpoint_LJ_TTS_PH_E2E_real_mel_snake_sigmoidur_R_newsine_conf_tvstyle_mask/val_loss_1.5.t7", map_location='cpu')
# params = torch.load("./checkpoint_LJ_TTS_PH_E2E_real_mel_snake_sigmoidur_R_newsine_conf_tvstyle_sup_codec_dis_codec_ft_cheat_dis/val_loss_tensor(1.8676, device='cuda:0').t7", map_location='cpu')
params = torch.load("./small_style_notext_robust_rvq_dis/val_loss_tensor(0.5101, device='cuda:0').t7", map_location='cpu')

opt = params['optimizer']
params = params['net']
for key in model:
    if key in params:
#         if not (key == "bert_encoder" or key == "predictor" or key == "bert"  or key == "msd" or key == "mpd"):
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
    #                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]
asr_model_copy = asr_model_copy.eval()
F0_model_copy = F0_model_copy.eval()
# model.prosody_encoder.load_state_dict(model.style_encoder.state_dict(), strict=False)

# model.predictor_encoder.load_state_dict(model.style_encoder.state_dict(), strict=False)


epochs = 100


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
        
for key in model:
    if key != "discriminator" and key != "dur_discriminator":
        model[key] = MyDataParallel(model[key])
        
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


class GeneratorLoss(torch.nn.Module):

    def __init__(self, discriminator):
        """Initilize spectral convergence loss module."""
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
        
    def forward(self, pred, real, h, mel_input_length, s2s_attn_mono):
        f_out, f_hidden = self.discriminator(pred, 
                      h.detach(), 
                      mel_input_length // (2 ** asr_model.n_down), 
                      s2s_attn_mono.size(-1))
        
        with torch.no_grad():
            r_out, r_hidden = self.discriminator(real.detach(), 
                          h.detach(), 
                          mel_input_length // (2 ** asr_model.n_down), 
                          s2s_attn_mono.size(-1))
        
        loss_gens = []
        loss_fms = []
        for bib in range(len(mel_input_length)):
            mel_len = mel_input_length[bib] // 2

            loss_gen = torch.mean((1-f_out[bib, :mel_len])**2) +\
                        generator_TPRLS_loss([r_out[bib, :mel_len].unsqueeze(0)], 
                                              [f_out[bib, :mel_len].unsqueeze(0)])
            
            loss_fm = 0
            for r, f in zip(r_hidden, f_hidden):
                loss_fm += F.l1_loss(r[bib, :mel_len], f[bib, :mel_len])

            if not torch.isnan(loss_gen):
                loss_gens.append(loss_gen)
            
            loss_fms.append(loss_fm)
            
        loss_gen = torch.stack(loss_gens).mean()
        loss_fm = torch.stack(loss_fms).mean()

        return loss_gen, loss_fm
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, discriminator):
        """Initilize spectral convergence loss module."""
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        
    def forward(self, pred, real, h, mel_input_length, s2s_attn_mono):
        f_out, _ = self.discriminator(pred.detach(), 
                      h.detach(), 
                      mel_input_length // (2 ** asr_model.n_down), 
                      s2s_attn_mono.size(-1))
        r_out, _ = self.discriminator(real.detach(), 
                      h.detach(), 
                      mel_input_length // (2 ** asr_model.n_down), 
                      s2s_attn_mono.size(-1))
    
    
        loss_diss = []
        for bib in range(len(mel_input_length)):
            mel_len = mel_input_length[bib] // 2

            loss_dis = torch.mean((f_out[bib, :mel_len])**2) +\
                        torch.mean((1-r_out[bib, :mel_len])**2) +\
                        discriminator_TPRLS_loss([r_out[bib, :mel_len].unsqueeze(0)], 
                                               [f_out[bib, :mel_len].unsqueeze(0)])
            if not torch.isnan(loss_dis):
                loss_diss.append(loss_dis)

        d_loss = torch.stack(loss_diss).mean()
        
        return d_loss
    
gl_p = GeneratorLoss(model.discriminator).to('cuda')
dl_p = DiscriminatorLoss(model.discriminator).to('cuda')
gl_p = MyDataParallel(gl_p)
dl_p = MyDataParallel(dl_p)

gl_d = GeneratorLoss(model.dur_discriminator).to('cuda')
dl_d = DiscriminatorLoss(model.dur_discriminator).to('cuda')
gl_d = MyDataParallel(gl_d)
dl_d = MyDataParallel(dl_d)


scheduler_params = {
    "max_lr": 1e-4,
    "pct_start": float(0),
    "epochs": epochs,
    "steps_per_epoch": len(train_dataloader),
}
scheduler_params_dict= {key: scheduler_params.copy() for key in model}
scheduler_params_dict['bert']['max_lr'] = 2e-5

optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict) 


for g in optimizer.optimizers['bert'].param_groups:
    g['betas'] = (0.9, 0.99)
    g['lr'] = 1e-5
    g['initial_lr'] = 1e-5
    g['min_lr'] = 0
    g['weight_decay'] = 0.01
    
optimizer.load_state_dict(opt)

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
loss_train_record = list([])
loss_test_record = list([])
iters = 0

log_interval = 10 # log every 10 iterations
saving_epoch = 1 # save every 5 epochs

criterion = nn.L1Loss() # F0 loss (regression)
torch.cuda.empty_cache()


from transforms import build_transforms, PitchShift, TimeStrech

class TimeStrech(nn.Module):
    def __init__(self, scale):
        super(TimeStrech, self).__init__()
        self.scale = scale

    def forward(self, x):
        mel_size = x.size(-1)
        
        x = F.interpolate(x, scale_factor=(1, self.scale), align_corners=False,
                          recompute_scale_factor=True, mode='bilinear').squeeze()
        
        return x.unsqueeze(1)
    
import time


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)

        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
stft_loss = MultiResolutionSTFTLoss().to('cuda')

to_mel_cuda = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300).to('cuda')

def to_mels(wave_tensor):
    mel_tensor = to_mel_cuda(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor) - mean) / std
    return mel_tensor

optimizer.optimizers['bert']


def shuffle_batch(batch):
    """
    Shuffles the elements within each component of the batch in unison.
    
    Parameters:
    *batch: Variable-length batch components, e.g., texts, input_lengths, mels, etc.
    
    Returns:
    A tuple of batch components shuffled in unison.
    """
    batch_size = len(batch[0])
    # Generate a random permutation of indices based on batch size
    indices = torch.randperm(batch_size)
    
    # Apply this permutation to each component of the batch
    shuffled_batch = [component[indices] for component in batch]
    
    return indices, tuple(shuffled_batch)


mypath  ="/local/data_cache/"

from os import listdir
from os.path import isfile, join

def merge_batches(batch1, batch2):
    batch1 = [b.squeeze() for b in batch1]
    batch2 = [b.squeeze() for b in batch2]

    waves, texts1, input_lengths1, mels1, mel_input_length1, adj_texts1, ref_lengths1, adj_mels1, adj_mels_lengths1 = batch1
    waves2, texts2, input_lengths2, mels2, mel_input_length2, adj_texts2, ref_lengths2, adj_mels2, adj_mels_lengths2 = batch2
    
    waves = [w for w in waves]
    waves2 = [w for w in waves2]
    waves.extend(waves2)
    
    nmels = mels1[0].size(0)
    max_text_length = max(input_lengths1.max(), input_lengths2.max())
    max_mel_length = max(mel_input_length1.max(), mel_input_length2.max())

    max_adjmel_length = max(adj_mels_lengths1.max(), adj_mels_lengths2.max())
    max_ref_length = max(ref_lengths1.max(), ref_lengths2.max())
    
    batch_size = mels1.size(0) + mels2.size(0)
    
    mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
    texts = torch.zeros((batch_size, max_text_length)).long()
    input_lengths = torch.zeros(batch_size).long()
    mel_input_length = torch.zeros(batch_size).long()
    adj_mels = torch.zeros((batch_size, nmels, max_adjmel_length)).float()
    adj_mels_lengths = torch.zeros(batch_size).long()
    ref_texts = torch.zeros((batch_size, max_ref_length)).long()
    ref_lengths = torch.zeros(batch_size).long()
    
    mels[:mels1.size(0), :, :mels1.size(-1)] = mels1
    mels[mels1.size(0):, :, :mels2.size(-1)] = mels2

    texts[:mels1.size(0), :texts1.size(-1)] = texts1
    texts[mels1.size(0):, :texts2.size(-1)] = texts2

    input_lengths[:mels1.size(0)] = input_lengths1
    input_lengths[mels1.size(0):] = input_lengths2

    mel_input_length[:mels1.size(0)] = mel_input_length1
    mel_input_length[mels1.size(0):] = mel_input_length2

    adj_mels[:adj_mels1.size(0), :, :adj_mels1.size(-1)] = adj_mels1
    adj_mels[adj_mels1.size(0):, :, :adj_mels2.size(-1)] = adj_mels2

    adj_mels_lengths[:mels1.size(0)] = adj_mels_lengths1
    adj_mels_lengths[mels1.size(0):] = adj_mels_lengths2

    ref_texts[:mels1.size(0), :adj_texts1.size(-1)] = adj_texts1
    ref_texts[mels1.size(0):, :adj_texts2.size(-1)] = adj_texts2

    ref_lengths[:mels1.size(0)] = ref_lengths1
    ref_lengths[mels1.size(0):] = ref_lengths2

    return waves, texts, input_lengths, mels, mel_input_length, ref_texts, ref_lengths, adj_mels, adj_mels_lengths


def random_mask_tokens(input_tensor, M, part=5):
    """
    Randomly mask tokens in the input tensor, ensuring at least M portion remains unmasked.

    Args:
    input_tensor (torch.Tensor): The input tensor of shape [512, T].
    M (float): The minimum portion of tokens that should remain unmasked.

    Returns:
    torch.Tensor: The masked input tensor.
    """
    B, T = input_tensor.shape

    if T <= M:
        return input_tensor
    masked_part = np.random.randint(0, part)
    
    max_mask = T - M
    masked_len = 0
    
    masked_tensor = input_tensor.clone()
    for i in range(masked_part):
        mask_start = np.random.randint(0, T)
        mask_end = np.random.randint(mask_start, T)
        
        if (mask_end - mask_start) + masked_len > max_mask:
            continue
            
        masked_tensor[:, mask_start:mask_end] = 0
        
        masked_len += (mask_end - mask_start)

    return masked_tensor

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

coll = 0

token_dict = {value: key for key, value in val_dataloader.dataset.text_cleaner.word_index_dictionary.items()}

state_queue = []

for epoch in range(epochs):
    running_loss = 0
    start_time = time.time()

    _ = [model[key].eval() for key in model]

    model.dur_predictor.train()
    model.pro_predictor.train()

    model.bert_encoder.train()
    model.bert.train()
    model.msd.train()
    model.mpd.train()
    
    i = 0
#     try: 
    while True:
#         waves_old = batch[0]
#         batch = [b.to(device) for b in batch[1:]]
#         indices, batch = shuffle_batch(batch)

#         waves = [waves_old[i] for i in indices]

#         texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch

        
#         onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#         cache_file = mypath + random.choice(onlyfiles)
#         batch = torch.load(cache_file)


#         onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#         cache_file1 = mypath + random.choice(onlyfiles)
#         if not os.path.isfile(cache_file1):
#             onlyfiles.remove(cache_file1.replace(mypath, ""))
#             coll += 1
#             continue
#         try:
#             batch1 = torch.load(cache_file1)
#         except:
#             os.remove(cache_file1)
#             continue
            
#         cache_file2 = mypath + random.choice(onlyfiles)
#         if not os.path.isfile(cache_file2):
#             onlyfiles.remove(cache_file2.replace(mypath, ""))
#             coll += 1
#             continue
#         try:
#             batch2 = torch.load(cache_file2)
#         except:
#             os.remove(cache_file2)
#             continue
        
#         batch = merge_batches(batch1, batch2)

        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        cache_file = mypath + random.choice(onlyfiles)
        try:
            batch = torch.load(cache_file)
        except:
            os.remove(cache_file)
            continue
            
        if len(onlyfiles) > 100 and (i + 1) % 50 != 0:
            os.remove(cache_file)

            waves = batch[0]
            batch = batch[1:]
            batch = [b.to(device) for b in batch]
            texts, input_lengths, mels, mel_input_length, ref_texts, ref_lengths, adj_mels, adj_mels_lengths = batch
        else:            
            _, batch = next(enumerate(train_dataloader))
            waves_old = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            indices, batch = shuffle_batch(batch)

            waves = [waves_old[i] for i in indices]

            texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch
        
        if input_lengths.max() > 512 or ref_lengths.max() > 512:
            print(input_lengths.max(), ref_lengths.max())
            continue
        
        if ((input_lengths / mel_input_length).max() > 0.7 or  (ref_lengths / adj_mels_lengths).max() > 0.7):
            print((input_lengths / mel_input_length).max(),
                  (ref_lengths / adj_mels_lengths).max())
            continue

        mels_new = []
        texts_new = []

        for bib in range(len(mel_input_length)):
            if (input_lengths[bib] + ref_lengths[bib]) > 512 or (mel_input_length[bib] + adj_mels_lengths[bib]) // 80 > 21:
                mels_new.append(mels[bib, :, :mel_input_length[bib]])
                texts_new.append(texts[bib, :input_lengths[bib]])
            else:
                text = ''.join([token_dict[int(k)] for k in texts[bib, :input_lengths[bib]]]).replace('$', '').strip()
                adj_text = ''.join([token_dict[int(k)] for k in ref_texts[bib, :ref_lengths[bib]]]).replace('$', '').strip()
                text = adj_text + " " + text
                text = val_dataloader.dataset.text_cleaner(text)
                text.insert(0, 0)
                text.append(0)
                text = torch.LongTensor(text)

                mel = torch.cat([adj_mels[bib, :, :adj_mels_lengths[bib]], 
                                mels[bib, :, :mel_input_length[bib]]
                                ], dim=-1)
                mels_new.append(mel)
                texts_new.append(text)

        mels = torch.zeros(mels.size(0), mels.size(1), max([m.size(-1) for m in mels_new]))
        texts  = torch.zeros(texts.size(0), max([m.size(-1) for m in texts_new]))
        
        for bib in range(len(mel_input_length)):
            mel_input_length[bib] = mels_new[bib].size(-1)
            input_lengths[bib] = texts_new[bib].size(-1)

            mels[bib, :, :mel_input_length[bib]] = mels_new[bib]
            texts[bib, :input_lengths[bib]] = texts_new[bib]
        mels = mels.to(adj_mels.device)
        texts = texts.to(adj_mels.device).long()
        
        if ((input_lengths / mel_input_length).max() > 0.7 or  (ref_lengths / adj_mels_lengths).max() > 0.7):
            print((input_lengths / mel_input_length).max(),
                  (ref_lengths / adj_mels_lengths).max())
            continue
                
        with torch.no_grad():
            mask = length_to_mask(adj_mels_lengths // (2 ** asr_model.n_down)).to('cuda')
            mel_mask = length_to_mask(adj_mels_lengths).to('cuda')
            text_mask = length_to_mask(ref_lengths).to(ref_texts.device)

            _, _, s2s_attn = asr_model(adj_mels, mask, ref_texts)
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            mask_ST = mask_from_lens(s2s_attn, ref_lengths, adj_mels_lengths // (2 ** asr_model.n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            real_norm = log_norm(adj_mels.unsqueeze(1)).squeeze(1)

            F0_real = F0_model(adj_mels.unsqueeze(1), adj_mels_lengths.cpu())

            real_norm.masked_fill_(mel_mask, 0.0)
            F0_real.masked_fill_(mel_mask, 0.0)

        position = torch.stack([torch.range(0, ref_texts.size(-1) - 1)] * ref_texts.size(0)).to('cuda')
        t_en = model.text_embedding(position.long()).transpose(-1, -2)

#         if np.random.rand() < 0.5:
#             asr = (t_en @ s2s_attn)
#         else:
#             asr = (t_en @ s2s_attn_mono)

        asr = (t_en @ s2s_attn_mono)
        asr_ref = torch.cat([asr, 
                                 F.avg_pool1d(real_norm, kernel_size=2).unsqueeze(1), 
                                 F.avg_pool1d(F0_real, kernel_size=2).unsqueeze(1)
                                ], axis=1)

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
            mel_mask = length_to_mask(mel_input_length).to('cuda')
            text_mask = length_to_mask(input_lengths).to(texts.device)

            _, _, s2s_attn = asr_model(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)

            F0_real = F0_model(mels.unsqueeze(1), mel_input_length.cpu())

            real_norm.masked_fill_(mel_mask, 0.0)
            F0_real.masked_fill_(mel_mask, 0.0)

        position = torch.stack([torch.range(0, texts.size(-1) - 1)] * texts.size(0)).to('cuda')
        t_en = model.text_embedding(position.long()).transpose(-1, -2)

#         if np.random.rand() < 0.5:
#             asr = (t_en @ s2s_attn)
#         else:
#             asr = (t_en @ s2s_attn_mono)

        asr = (t_en @ s2s_attn_mono)
        asr_real = torch.cat([asr, 
                                 F.avg_pool1d(real_norm, kernel_size=2).unsqueeze(1), 
                                 F.avg_pool1d(F0_real, kernel_size=2).unsqueeze(1)
                                ], axis=1)
        
        
        if bool(random.getrandbits(1)):
#             # compute prosodic style
#             if bool(random.getrandbits(1)):
#                 min_length = (adj_mels_lengths // 2).min()
#                 asr_new = torch.zeros_like(asr_ref).to(asr_ref.device)
#                 bid = 0
#                 for a, length in zip (asr_ref, adj_mels_lengths // 2):
#                     a = a[:, :length]
#                     asr_m = random_mask_tokens(a, min_length)
#                     asr_new[bid, :, :length] = asr_m
#                     bid += 1
#                 h_cont = model.predictor_encoder(asr_ref, adj_mels_lengths // 2)
#             else:
            min_length = (mel_input_length // 2).min()
            asr_new = torch.zeros_like(asr_real).to(asr_real.device)
            bid = 0
            for a, length in zip (asr_real, mel_input_length // 2):
                a = a[:, :length]
                asr_m = random_mask_tokens(a, min_length)
                asr_new[bid, :, :length] = asr_m
                bid += 1
            h_cont = model.predictor_encoder(asr_new, mel_input_length // 2)
        else:
            flag = False
            mel_len = min([int(mel_input_length.min().item() / 2 - 1)])
            mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
            st = []
            gt = []
            for bib in range(len(adj_mels_lengths)):

                mel_length = int(adj_mels_lengths[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(asr_ref[bib, :, (random_start):((random_start+mel_len_st))])

                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                gt.append(asr_real[bib, :, (random_start):((random_start+mel_len))])

            st = torch.stack(st).detach()
            gt = torch.stack(gt).detach()

            if random.random() < 0.9:
                st = gt
                
            h_cont = model.predictor_encoder(st, (torch.ones(st.size(0)).to(st.device) * st.size(-1)).long())
        h, commitment_loss, codebook_loss, code = model.quantizer(h_cont)
        commitment_loss, codebook_loss = commitment_loss.mean(), codebook_loss.mean()
    
        with torch.no_grad():
            mel_len = int(mel_input_length.min().item() / 2 - 1)
            gt = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

            gt = torch.stack(gt).detach()

        # bert feature
        bert_dur = model.bert(texts, attention_mask=(~text_mask).int()).last_hidden_state
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

#         s, d_en = model.prosody_encoder(gt, d_en, input_lengths, t_en.size(-1))
        
#         h = torch.cat([h_cont, s.unsqueeze(-1)], dim=-1)
        
        # predict duration
        d = model.dur_predictor(d_en, h, input_lengths, d_en.size(-1))
        d.masked_fill_(text_mask.unsqueeze(-1), 0.0)


        # predict prosody
        F0_fake, N_fake = model.pro_predictor((d_en @ s2s_attn_mono), 
                                              h, 
                                              mel_input_length // (2 ** asr_model.n_down), 
                                              s2s_attn_mono.size(-1))

        F0_fake.masked_fill_(mel_mask, 0.0)
        N_fake.masked_fill_(mel_mask, 0.0)
        
        dur = torch.sigmoid(d).sum(axis=-1)
        dur.masked_fill_(text_mask, 0.0)
#         d_pred = dur + dur.round().detach() - dur.detach()
        d_pred = dur

        dur_real = torch.cat([t_en, 
                            d_gt.unsqueeze(1)
                                    ], axis=1)
        dur_pred = torch.cat([t_en, 
                            d_pred.unsqueeze(1)
                                    ], axis=1)

        with torch.no_grad():
            asr_pred = torch.cat([asr, 
                                 F.avg_pool1d(N_fake, kernel_size=2).unsqueeze(1), 
                                 F.avg_pool1d(F0_fake, kernel_size=2).unsqueeze(1)
                                ], axis=1)
            h_pred = model.predictor_encoder(asr_pred, mel_input_length // 2)

            loss_sty = F.l1_loss(h_cont.detach(), h_pred)



        optimizer.zero_grad()
        d_loss = dl_p(asr_pred, asr_real, h, mel_input_length, s2s_attn_mono).mean()
        d_loss += dl_d(dur_pred, dur_real, h, input_lengths, dur_pred).mean()
        d_loss.backward()
        optimizer.step('discriminator')
        optimizer.step('dur_discriminator')
            
        optimizer.zero_grad()

        loss_s2s = 0
        loss_algn = 0
        for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
            _s2s_pred = _s2s_pred[:_text_length, :]
            _text_input = _text_input[:_text_length].long()
            _s2s_trg = torch.zeros_like(_s2s_pred)
            for p in range(_s2s_trg.shape[0]):
                _s2s_trg[p, :_text_input[p]] = 1
#             _dur_pred = F.relu(torch.sigmoid(_s2s_pred) - 0.5).sum(axis=1) * 2
            _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

            loss_algn += F.l1_loss(_dur_pred[1:_text_length-1], 
                                   _text_input[1:_text_length-1])
            loss_s2s += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

        loss_s2s /= texts.size(0)
        loss_s2s *= 20
        loss_algn /= texts.size(0)

        loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10 * ((real_norm.size(-1) * batch_size) / (mel_input_length).sum())
        loss_norm_rec = F.smooth_l1_loss(real_norm, N_fake) * ((real_norm.size(-1) * batch_size) / (mel_input_length).sum())

        loss_gen1, loss_fm1 = gl_p(asr_pred, asr_real, h, mel_input_length, s2s_attn_mono)
        loss_gen2, loss_fm2 = gl_d(dur_pred, dur_real, h, input_lengths, dur_pred)

        loss_gen = (loss_gen1.mean() + loss_gen2.mean())  + (loss_fm1.mean() + loss_fm2.mean())


        g_loss = loss_F0_rec + loss_s2s + loss_norm_rec + loss_algn + commitment_loss + codebook_loss + loss_gen + loss_sty # + loss_gen * 0.1

        g_loss.backward()

#         optimizer.step('bert_encoder')
#         optimizer.step('bert')
        optimizer.step('dur_predictor')
        optimizer.step('pro_predictor')
        optimizer.step('predictor_encoder')
        optimizer.step('quantizer')
        optimizer.step('text_embedding')

        running_loss += loss_F0_rec.item()
                
        i += 1
                
        iters = iters + 1
        if (i+1)%log_interval == 0:
            print ('Epoch [%d/%d], Step [%d/%d], F0 Loss: %.5f, Algn Loss: %.5f, Norm Loss: %.5f, S2S Loss: %.5f, Sty Loss: %.5f, Dis Loss: %.5f, Gen Loss: %.5f, Comm Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_dataloader), running_loss / log_interval, loss_algn, loss_norm_rec, loss_s2s, loss_sty, d_loss, loss_gen, commitment_loss))
            running_loss = 0
            print('Time elasped:', time.time()-start_time)


    loss_test = 0
    loss_align = 0
    loss_f = 0

    _ = [model[key].eval() for key in model]

    with torch.no_grad():
        iters_test = 0
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch
                batch_size = len(waves)
                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
                    mel_mask = length_to_mask(mel_input_length).to('cuda')
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    try:
                        _, _, s2s_attn = asr_model(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)
                    except:
                        continue

                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                    position = torch.stack([torch.range(0, texts.size(-1) - 1)] * texts.size(0)).to('cuda')
                    t_en = model.text_embedding(position.long()).transpose(-1, -2)

#                     mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
#                     st = []
#                     for bib in range(len(adj_mels_lengths)):
#                         mel_length = int(adj_mels_lengths[bib].item() / 2)
#                         random_start = np.random.randint(0, mel_length - mel_len_st)
#                         st.append(adj_mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

#                     st = torch.stack(st).detach()

#                     s, t_en = model.prosody_encoder(st, t_en, input_lengths, t_en.size(-1))

                    asr = (t_en @ s2s_attn)

                    # encode
                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)

                    F0_real = F0_model(mels.unsqueeze(1), mel_input_length.cpu())

                    real_norm.masked_fill_(mel_mask, 0.0)
                    F0_real.masked_fill_(mel_mask, 0.0)
                    
                    

                    asr = torch.cat([asr, 
                                         F.avg_pool1d(real_norm, kernel_size=2).unsqueeze(1), 
                                         F.avg_pool1d(F0_real, kernel_size=2).unsqueeze(1)
                                        ], axis=1)


                # compute prosodic style
                h = model.predictor_encoder(asr, mel_input_length // 2)
                h, commitment_loss, codebook_loss, code = model.quantizer(h)
                
                # bert feature
                bert_dur = model.bert(texts, attention_mask=(~text_mask).int()).last_hidden_state
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

                # predict duration
                d = model.dur_predictor(d_en, h, input_lengths, d_en.size(-1))
                d.masked_fill_(text_mask.unsqueeze(-1), 0.0)


                # predict prosody
                F0_fake, N_fake = model.pro_predictor((d_en @ s2s_attn_mono), 
                                                      h, 
                                                      mel_input_length // (2 ** asr_model.n_down), 
                                                      s2s_attn_mono.size(-1))

                F0_fake.masked_fill_(mel_mask, 0.0)
                N_fake.masked_fill_(mel_mask, 0.0)

                loss_s2s = 0
                loss_algn = 0
                for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                    _s2s_pred = _s2s_pred[:_text_length, :]
                    _text_input = _text_input[:_text_length].long()
                    _s2s_trg = torch.zeros_like(_s2s_pred)
                    for p in range(_s2s_trg.shape[0]):
                        _s2s_trg[p, :_text_input[p]] = 1
        #             _dur_pred = F.relu(torch.sigmoid(_s2s_pred) - 0.5).sum(axis=1) * 2
                    _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                    loss_algn += F.l1_loss(_dur_pred[1:_text_length-1], 
                                           _text_input[1:_text_length-1])
                    loss_s2s += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

                loss_s2s /= texts.size(0)
                loss_s2s *= 20
                loss_algn /= texts.size(0)

                loss_F0 =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10 * ((real_norm.size(-1) * batch_size) / (mel_input_length).sum())
                loss_norm_rec = F.smooth_l1_loss(real_norm, N_fake) * ((real_norm.size(-1) * batch_size) / (mel_input_length).sum())

                loss_test += loss_F0
                loss_align += loss_algn
                loss_f += loss_norm_rec
                iters_test += 1
                print(loss_F0, loss_algn, loss_norm_rec)
            except:
                continue
        print('Epochs:', epoch + 1)
        print('Validation loss: %.3f, %.3f, %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test), '\n\n\n')

        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            if not os.path.isdir('small_style_notext_robust_rvq_dis_half'):
                os.mkdir('small_style_notext_robust_rvq_dis_half')
            torch.save(state, './small_style_notext_robust_rvq_dis_half/val_loss_' + str((loss_test / iters_test)) + '.t7')
#     except:
#         pass
    loss_test = 0
    loss_align = 0
    loss_f = 0

    _ = [model[key].eval() for key in model]

    with torch.no_grad():
        iters_test = 0
        for batch_idx, batch in enumerate(val_dataloader):
            # try:
            #     optimizer.zero_grad()

            #     waves = batch[0]
            #     batch = [b.to(device) for b in batch[1:]]
            #     texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch
            #     batch_size = len(waves)
            #     with torch.no_grad():
            #         mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
            #         mel_mask = length_to_mask(mel_input_length).to('cuda')
            #         text_mask = length_to_mask(input_lengths).to(texts.device)

            #         try:
            #             _, _, s2s_attn = asr_model(mels, mask, texts)
            #             s2s_attn = s2s_attn.transpose(-1, -2)
            #             s2s_attn = s2s_attn[..., 1:]
            #             s2s_attn = s2s_attn.transpose(-1, -2)
            #         except:
            #             continue

            #         mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
            #         s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            #         position = torch.stack([torch.range(0, texts.size(-1) - 1)] * texts.size(0)).to('cuda')
            #         t_en = model.text_embedding(position.long()).transpose(-1, -2)

#                     mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
#                     st = []
#                     for bib in range(len(adj_mels_lengths)):
#                         mel_length = int(adj_mels_lengths[bib].item() / 2)
#                         random_start = np.random.randint(0, mel_length - mel_len_st)
#                         st.append(adj_mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

#                     st = torch.stack(st).detach()

#                     s, t_en = model.prosody_encoder(st, t_en, input_lengths, t_en.size(-1))

                    asr = (t_en @ s2s_attn)
                           
#         mels_new = []
#         texts_new = []
#         d_gt_new = []

#         for bib in range(len(mel_input_length)):
#             if (input_lengths[bib] + ref_lengths[bib]) > 512:
#                 mels_new.append(mels[bib])
#                 texts_new.append(texts[bib])
#             else:
#                 text = ''.join([token_dict[int(k)] for k in texts[bib, 1:input_lengths[bib] - 1]])
#                 adj_text = ''.join([token_dict[int(k)] for k in ref_texts[bib, 1:ref_lengths[bib] - 1]])
#                 d = d_gt[bib, 1:input_lengths[bib] - 1]
#                 adj_d = d_gt_adj[bib, 1:ref_lengths[bib] - 1]
                
#                 text = adj_text + text
#                 text = val_dataloader.dataset.text_cleaner(text)
#                 text.insert(0, 0)
#                 text.append(0)
#                 text = torch.LongTensor(text)
                
#                 dd = torch.cat([d_gt[bib, 0].unsqueeze(0), adj_d, d, d_gt[bib, 1].unsqueeze(0)])

#                 mel = torch.cat([adj_mels[bib, :, :adj_mels_lengths[bib]], 
#                                 mels[bib, :, :mel_input_length[bib]]
#                                 ], dim=-1)
#                 mels_new.append(mel)
#                 texts_new.append(text)
#                 d_gt_new.append(dd)

#         mels = torch.zeros(mels.size(0), mels.size(1), max([m.size(-1) for m in mels_new]))
#         texts  = torch.zeros(texts.size(0), max([m.size(-1) for m in texts_new]))
#         d_gt  = torch.zeros(texts.size(0), max([m.size(-1) for m in texts_new]))

#         for bib in range(len(mel_input_length)):
#             mel_input_length[bib] = mels_new[bib].size(-1)
#             input_lengths[bib] = texts_new[bib].size(-1)

#             mels[bib, :, :mel_input_length[bib]] = mels_new[bib]
#             texts[bib, :input_lengths[bib]] = texts_new[bib]
#             d_gt[bib, :input_lengths[bib]] = dd[bib]

            
#         mels = mels.to(adj_mels.device)
#         texts = texts.to(adj_mels.device).long()
#         d_gt = d_gt.to(adj_mels.device).long()

_ = [model[key].eval() for key in model]

idx = 1


with torch.no_grad():
    mask = length_to_mask(adj_mels_lengths // (2 ** asr_model.n_down)).to('cuda')
    mel_mask = length_to_mask(adj_mels_lengths).to('cuda')
    text_mask = length_to_mask(ref_lengths).to(ref_texts.device)
    
    _, _, s2s_attn = asr_model(adj_mels, mask, ref_texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)

    mask_ST = mask_from_lens(s2s_attn, ref_lengths, adj_mels_lengths // (2 ** asr_model.n_down))
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

    # encode
    d_gt = s2s_attn_mono.sum(axis=-1).detach()

    real_norm = log_norm(adj_mels.unsqueeze(1)).squeeze(1)

    F0_real = F0_model(adj_mels.unsqueeze(1), adj_mels_lengths.cpu())

    real_norm.masked_fill_(mel_mask, 0.0)
    F0_real.masked_fill_(mel_mask, 0.0)

    position = torch.stack([torch.range(0, ref_texts.size(-1) - 1)] * ref_texts.size(0)).to('cuda')
    t_en = model.text_embedding(position.long()).transpose(-1, -2)

    asr = (t_en @ s2s_attn_mono)
    asr_ref = torch.cat([asr, 
                             F.avg_pool1d(real_norm, kernel_size=2).unsqueeze(1), 
                             F.avg_pool1d(F0_real, kernel_size=2).unsqueeze(1)
                            ], axis=1)
    
with torch.no_grad():
    mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
    mel_mask = length_to_mask(mel_input_length).to('cuda')
    text_mask = length_to_mask(input_lengths).to(texts.device)

    _, _, s2s_attn = asr_model(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)

    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

    # encode
    d_gt = s2s_attn_mono.sum(axis=-1).detach()

    real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)

    F0_real = F0_model(mels.unsqueeze(1), mel_input_length.cpu())

    real_norm.masked_fill_(mel_mask, 0.0)
    F0_real.masked_fill_(mel_mask, 0.0)

    position = torch.stack([torch.range(0, texts.size(-1) - 1)] * texts.size(0)).to('cuda')
    t_en = model.text_embedding(position.long()).transpose(-1, -2)

    asr = (t_en @ s2s_attn_mono)
    asr_real = torch.cat([asr, 
                             F.avg_pool1d(real_norm, kernel_size=2).unsqueeze(1), 
                             F.avg_pool1d(F0_real, kernel_size=2).unsqueeze(1)
                            ], axis=1)
    
with torch.no_grad():
    mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
    mel_len = min([int(mel_input_length.min().item() / 2 - 1)])

    st = []
    gt = []
    for bib in range(len(adj_mels_lengths)):
        mel_length = int(adj_mels_lengths[bib].item() / 2)
        random_start = np.random.randint(0, mel_length - mel_len_st)
        st.append(adj_mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])

        mel_length = int(mel_input_length[bib].item() / 2)
        random_start = np.random.randint(0, mel_length - mel_len)
        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

    st = torch.stack(st).detach()
    gt = torch.stack(gt).detach()

with torch.no_grad():
    mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
    ppgs, s2s_pred, s2s_attn = asr_model(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    text_mask = length_to_mask(input_lengths).to(texts.device)
    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
    attn_mask = (attn_mask < 1)
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
    s2s_attn.masked_fill_(attn_mask, 0.0)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    # encode
    t_en = model.text_encoder(texts, input_lengths, text_mask)
    asr = (t_en @ s2s_attn_mono)
    with torch.no_grad():
        F0_down = 7
        F0_real = F0_model(mels.unsqueeze(1), mel_input_length.cpu())
        F0_real = nn.functional.conv1d(F0_real.unsqueeze(1), torch.ones(1, 1, F0_down).to('cuda'), padding=F0_down//2).squeeze(1) / F0_down
        
    real_norm = log_norm(mels[idx, :, :mel_input_length[idx]].unsqueeze(0).unsqueeze(1)).squeeze(1)
    # reconstruct
    dix = idx
    
    with torch.no_grad():
        F0_ref = F0_model(mels[(dix-1) % batch_size, :, :mel_input_length[(dix-1) % batch_size]].unsqueeze(0).unsqueeze(1),
                                mel_input_length[(dix-1) % batch_size].unsqueeze(0).cpu())
        N_ref = log_norm(mels[(dix-1) % batch_size, :, :mel_input_length[(dix-1) % batch_size]].unsqueeze(0).unsqueeze(1)).squeeze(1)
        
        F0_ref_median = F0_ref.median()
        F0_trg = F0_real[idx, :mel_input_length[idx]].unsqueeze(0)
        F0_trg = F0_trg / F0_trg.median() * F0_ref_median
        
        N_ref_mean = N_ref.median()
        N_trg = real_norm / real_norm.median() * N_ref_mean
    
#     st, t_en, input_lengths, t_en.size(-1)
    
    s_trg, t_en_trg = model.style_encoder(mels[(dix-1) % batch_size, :, :240 ].unsqueeze(0), 
                                          t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    
    asr_trg = (t_en_trg @ s2s_attn_mono[idx, ...].unsqueeze(0))
    
#     s_trg = torch.nn.functional.normalize(s_trg)
    # s_trg = model.style_encoder(ref_mels[idx, ...].unsqueeze(0).unsqueeze(1), torch.LongTensor([109]).to('cuda').unsqueeze(0)).squeeze(1)
    mel_fake = model.decoder(asr_trg[:, :, :mel_input_length[idx] // (2 ** asr_model.n_down)], 
                                F0_trg, N_trg,
#                             F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                                s_trg)
    
#     h = model.predictor_encoder(asr_ref, adj_mels_lengths // 2)
#     h = model.predictor_encoder(asr_real[..., 0:0 + 120], (torch.ones(asr_real.shape[0]).to('cuda') * 120).int())
    h = model.predictor_encoder(asr_real, mel_input_length // 2)
#     h_null = model.predictor_encoder(torch.zeros_like(asr_ref).to('cuda'), adj_mels_lengths // 2)
    h, commitment_loss, codebook_loss, code = model.quantizer(h)
#     h = model.quantizer.decode(code.transpose(-1, -2))
    
    # bert feature
    bert_dur = model.bert(texts, attention_mask=(~text_mask).int()).last_hidden_state
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
    
#     F0_fake, N_fake = model.pro_predictor((d_en @ s2s_attn_mono)[idx, :, :mel_input_length[idx] // (2 ** asr_model.n_down)].unsqueeze(0), 
#                                         h[(dix-1) % batch_size].unsqueeze(0), 
#                                         mel_input_length[idx].unsqueeze(0) // (2 ** asr_model.n_down), 
#                                     int(mel_input_length[idx] // (2 ** asr_model.n_down)))
    
    d = model.dur_predictor(d_en[idx, :, :input_lengths[idx]].unsqueeze(0), 
                                        h[(dix-1) % batch_size].unsqueeze(0), 
                                        input_lengths[idx].unsqueeze(0), 
                                    int(input_lengths[idx]))
    
    duration = torch.sigmoid(d).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    pred_aln_trg = torch.zeros(input_lengths[idx], int(pred_dur.sum().data)).to('cuda')
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
        c_frame += int(pred_dur[i].data)
        
    asr_fake = (t_en_trg[:, :, :input_lengths[idx]] @ pred_aln_trg)
        
#     F0_fake_null, N_fake_null = model.pro_predictor((d_en[idx, :, :input_lengths[idx]].unsqueeze(0) @ pred_aln_trg), 
#                                         h_null[(dix-1) % batch_size].unsqueeze(0), 
#                                         torch.LongTensor([pred_aln_trg.size(-1)]).to('cuda'), 
#                                     int(pred_aln_trg.size(-1)))
    F0_fake, N_fake = model.pro_predictor((d_en[idx, :, :input_lengths[idx]].unsqueeze(0) @ pred_aln_trg), 
                                        h[(dix-1) % batch_size].unsqueeze(0), 
                                        torch.LongTensor([pred_aln_trg.size(-1)]).to('cuda'), 
                                    int(pred_aln_trg.size(-1)))
    F0_fake = F.relu(F0_fake)
    
#     with torch.no_grad():
#         F0_ref_median = F0_fake.median()
#         F0_trg = F0_fake_null
#         F0_trg = F0_trg / F0_trg.median() * F0_ref_median
        
#         N_ref_mean = N_fake.median()
#         N_trg = N_fake_null / N_fake_null.median() * N_ref_mean
    
    
#     F0_fake, N_fake = F0_trg, N_trg
    
    mel_fake_pred = model.decoder(asr_fake, 
                                F0_fake, N_fake,
#                             F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                                s_trg)
    
    s, t_en_fake = model.style_encoder(adj_mels[idx, :, :adj_mels_lengths[idx] ].unsqueeze(0), 
                                t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    asr = (t_en_fake @ s2s_attn_mono[idx, ...].unsqueeze(0))
    F0_fake, N_fake = model.pro_predictor((d_en @ s2s_attn_mono)[idx, :, :mel_input_length[idx] // (2 ** asr_model.n_down)].unsqueeze(0), 
                                        h[idx].unsqueeze(0), 
                                        mel_input_length[idx].unsqueeze(0) // (2 ** asr_model.n_down), 
                                    int(mel_input_length[idx] // (2 ** asr_model.n_down)))
    F0_fake = F.relu(F0_fake)
    y_rec_fake = model.decoder(asr[:, :, :mel_input_length[idx] // 2], 
                            F0_fake, N_fake, 
                          s)
    
    s, t_en = model.style_encoder(mels[idx, :, :mel_input_length[idx] ].unsqueeze(0), 
                                t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    
    asr = (t_en @ s2s_attn_mono[idx, ...].unsqueeze(0))
    
    y_rec = model.decoder(asr[:, :, :mel_input_length[idx] // 2], 
                            F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                          s)
    # predict
    
    d = model.dur_predictor(d_en[idx, :, :input_lengths[idx]].unsqueeze(0), 
                                        h[idx].unsqueeze(0), 
                                        input_lengths[idx].unsqueeze(0), 
                                    int(input_lengths[idx]))
    
    duration = torch.sigmoid(d).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    pred_aln_trg = torch.zeros(input_lengths[idx], int(pred_dur.sum().data)).to('cuda')
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
        c_frame += int(pred_dur[i].data)
        
    asr_fake = (t_en[:, :, :input_lengths[idx]] @ pred_aln_trg)
        
    F0_fake, N_fake = model.pro_predictor((d_en[idx, :, :input_lengths[idx]].unsqueeze(0) @ pred_aln_trg), 
                                        h[idx].unsqueeze(0), 
                                        torch.LongTensor([pred_aln_trg.size(-1)]).to('cuda'), 
                                    int(pred_aln_trg.size(-1)))
    F0_fake = F.relu(F0_fake)
    
    mel_pred = model.decoder(asr_fake, 
                                F0_fake, N_fake,
#                             F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                                s)
    
    sig = 1.5
    output_lengths = []
    attn_preds = []
        
    # differentiable duration modeling
    for _s2s_pred, _text_length in zip(d, input_lengths[idx].unsqueeze(0)):
        _s2s_pred_org = _s2s_pred[:_text_length, :]
        _s2s_pred = torch.sigmoid(_s2s_pred_org)
        _dur_pred = _s2s_pred.sum(axis=-1)
        l = int(torch.round(_s2s_pred.sum()).item())
        t = torch.arange(0, l).expand(l)
        t = torch.arange(0, l).unsqueeze(0).expand((len(_s2s_pred), l)).to(texts.device)
        loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2
        f = torch.exp(-0.5 * torch.square(t - (l - loc.unsqueeze(-1))) / (sig)**2)
        out = torch.nn.functional.conv1d(_s2s_pred_org.unsqueeze(0), 
                                     f.unsqueeze(1), 
                                     padding=f.shape[-1] - 1, groups=int(_text_length))[..., :l]
        attn_preds.append(F.softmax(out.squeeze(), dim=0))
        output_lengths.append(l)
    max_len = max(output_lengths)
        
    attn_pred = torch.stack(attn_preds)
    
    asr_fake = (t_en[:, :, :input_lengths[idx]] @ attn_pred)
        
    F0_fake, N_fake = model.pro_predictor((d_en[idx, :, :input_lengths[idx]].unsqueeze(0) @ attn_pred), 
                                        h[idx].unsqueeze(0), 
                                        torch.LongTensor([attn_pred.size(-1)]).to('cuda'), 
                                    int(attn_pred.size(-1)))
    F0_fake = F.relu(F0_fake)
    mel_pred_diff = model.decoder(asr_fake, 
                                F0_fake, N_fake,
#                             F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                                s)
    
token_dict = {value: key for key, value in val_dataloader.dataset.text_cleaner.word_index_dictionary.items()}


''.join([token_dict[int(k)] for k in texts[idx, :input_lengths[idx]]])

out = mel_fake
