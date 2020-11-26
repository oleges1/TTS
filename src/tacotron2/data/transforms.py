from IPython import display
from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa
from matplotlib import pyplot as plt
import numpy as np


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251
    frequency_mask_max_percentage: float = 0.15
    time_mask_max_percentage: float = 0.0
    mask_probability: float = 0.3


class MaskSpectrogram(object):
    """Masking a spectrogram aka SpecAugment."""

    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, probability=1.0, pad_value=0.0):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = probability
        self.pad_value = pad_value

    def __call__(self, data):
        # for i in range(len(data['audio'])):
        if len(data.shape) == 2:
            if random.random() < self.probability:
                nu, tau = data.shape

                f = random.randint(0, int(self.frequency_mask_probability*nu))
                f0 = random.randint(0, nu - f)
                data[f0:f0 + f, :] = self.pad_value

                t = random.randint(0, int(self.time_mask_probability*tau))
                t0 = random.randint(0, tau - t)
                data[:, t0:t0 + t] = self.pad_value
        else:
            for i in range(len(data)):
                if random.random() < self.probability:
                    nu, tau = data[i].shape

                    f = random.randint(0, int(self.frequency_mask_probability*nu))
                    f0 = random.randint(0, nu - f)
                    data[i][f0:f0 + f, :] = self.pad_value

                    t = random.randint(0, int(self.time_mask_probability*tau))
                    t0 = random.randint(0, tau - t)
                    data[i][:, t0:t0 + t] = self.pad_value

        return data


class MelSpectrogram(nn.Module):

    def __init__(self, config=MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

        # self.mask_spec = MaskSpectrogram(
        #     frequency_mask_max_percentage=config.frequency_mask_max_percentage,
        #     time_mask_max_percentage=config.time_mask_max_percentage,
        #     probability=config.mask_probability,
        #     pad_value=config.pad_value
        # )


    def forward(self, data):
        mels = []
        for i in range(len(data['audio'])):
            mel = self.mel_spectrogram(data['audio'][i]) \
                .clamp_(min=1e-5) \
                .log_()

            # if self.training:
            #     mel = self.mask_spec(mel)

            mels.append(mel)
        data['mel'] = mels
        return data


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            try:
              data = t(data)
            except TypeError:
              # audiomentation transform
              data['audio'] = t(data['audio'], sample_rate=data['sample_rate'])
        return data


class AudioSqueeze:
    def __call__(self, data):
        data['audio'] = data['audio'].squeeze(0)
        return data


class TextPreprocess:
    def __init__(self, alphabet='SE!,;.? '):
        self.alphabet = alphabet
        self.sym2id = {a: i for i, a in enumerate(alphabet)}
        self.id2sym = {i: a for i, a in enumerate(alphabet)}

    def __call__(self, data):
        data['text'] = list(map(lambda x: self.sym2id.get(x, 0), list(data['text'].lower().strip()))) + [1]
        return data

    def reverse(self, vector):
        return ''.join([self.id2sym[x] for x in vector])

class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, data):
        data['audio'] = np.array(data['audio'])
        data['text'] = np.array(data['text'])
        return data

class AddLengths:
    def __call__(self, data):
        data['audio_lengths'] = torch.tensor([item.shape[-1] for item in data['audio']]).to(data['audio'][0].device)
        data['mel_lengths'] = torch.tensor([item.shape[0] for item in data['mel']]).to(data['mel'][0].device)
        data['text_lengths'] = torch.tensor([item.shape[0] for item in data['text']]).to(data['text'][0].device)
        return data

class ToGpu:
    def __init__(self, device):
        self.device = device

    def __call__(self, data):
        data = {k: [torch.from_numpy(np.array(item)).to(self.device) for item in v] for k, v in data.items()}
        return data

class Pad:
    def __init__(self, config=MelSpectrogramConfig):
        self.config = config

    def __call__(self, data):
        padded_batch = {}
        for k, v in data.items():
            padding_value = self.config.pad_value if k == 'audio' else 0
            if len(v[0].shape) < 2:
                items = [item[..., None] for item in v]
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(items,
                    batch_first=True, padding_value=padding_value)[..., 0]
            else:
                items = [item.permute(1, 0) for item in v]
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(items,
                    batch_first=True, padding_value=padding_value).permute(0, 2, 1)
        return padded_batch
