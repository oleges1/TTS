import torch
from torch import nn
from data.transforms import MelSpectrogram, MelSpectrogramConfig


class OptimizerVocoder():
    def __init__(self):
        self.mel = MelSpectrogram()
        self.min_level_db = -80.0
        self.ref_level_db = 20.0

    def get_mel(self, audio):
        return self.mel.mel_spectrogram(audio)

    def post_spec(self, x):
        x = (x - 1) * -self.min_level_db + self.ref_level_db
        x = torch.pow(10, x / 10)
        return x

    def pre_spec(self, x):
        x = torch.log10(x) * 10
        x = (x - self.ref_level_db) / -self.min_level_db + 1
        return x


    def inference(self, melspectrogram, iters=1000):
        x = torch.normal(0, 1e-6, size=((melspectrogram.size(-1) - 1) * MelSpectrogramConfig.hop_length, )).to(melspectrogram.device).requires_grad_()
        optimizer = torch.optim.LBFGS([x], tolerance_change=1e-16)
        criterion = torch.nn.MSELoss()
        melspectrogram = torch.clamp(melspectrogram, max=1e3)
        melspectrogram = self.post_spec(torch.exp(melspectrogram))

        def closure():
            optimizer.zero_grad()
            mel = self.get_mel(x)
            loss = criterion(mel, melspectrogram[0])
            loss.backward()
            return loss

        for i in range(iters):
            optimizer.step(closure=closure)
        return x.detach()[:, None]
