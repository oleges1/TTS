import torch
from data.transforms import MelSpectrogram, MelSpectrogramConfig


class OptimizerVocoder():
    def __init__(self):
        self.mel = MelSpectrogram()

    def get_mel(self, audio):
        return self.mel.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

    def inference(self, melspectrogram, iters=1000):
        x = torch.normal(0, 1e-6, size=((melspectrogram.size(1) - 1) * MelSpectrogramConfig.hop_length, )).to(melspectrogram.device).requires_grad_()
        optimizer = torch.optim.LBFGS([x], tolerance_change=1e-16)

        def closure():
            optimizer.zero_grad()
            mel = self.get_mel(x)
            loss = self.criterion(mel, melspectrogram)
            loss.backward()
            return loss

        for i in range(iters):
            optimizer.step(closure=closure)
        return x
