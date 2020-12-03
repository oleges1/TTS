import torch
import numpy as np
import random

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized


def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform


def mu_law_encode_torch(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = torch.tensor(quantization_channels - 1, device=audio.device, dtype=torch.float)
    quantize_space = torch.linspace(-1, 1, quantization_channels)

    quantized = torch.sign(audio) * torch.log(1 + mu * torch.abs(audio)) / torch.log(mu + 1)
    quantized = torch.bucketize(quantized, quantize_space) - 1

    return quantized


def mu_law_decode_torch(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = torch.tensor(quantization_channels - 1, device=output.device, dtype=torch.float)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = torch.sign(expanded) * (
                   torch.exp(torch.abs(expanded) * torch.log(mu + 1)) - 1
               ) / mu

    return waveform
