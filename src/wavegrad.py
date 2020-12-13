import torch
from torch import nn
from data.transforms import MelSpectrogram, MelSpectrogramConfig


class OptimizerVocoder():
    def __init__(self):
        self.mel = MelSpectrogram()

    def get_mel(self, audio):
        return self.mel.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

    def inference(self, melspectrogram, iters=1000):
        x = torch.normal(0, 1e-6, size=((melspectrogram.size(-1) - 1) * MelSpectrogramConfig.hop_length, )).to(melspectrogram.device).requires_grad_()
        optimizer = torch.optim.LBFGS([x], tolerance_change=1e-16)
        criterion = torch.nn.MSELoss()

        def closure():
            optimizer.zero_grad()
            mel = self.get_mel(x)
            loss = criterion(mel, melspectrogram[0])
            loss.backward()
            return loss

        for i in range(iters):
            optimizer.step(closure=closure)
        return x.detach()[:, None]
