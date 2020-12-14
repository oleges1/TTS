import torch
from torch import nn
import librosa
from data.transforms import MelSpectrogram, MelSpectrogramConfig


class OptimizerVocoder():
    def __init__(self):
        self.mel = MelSpectrogram()

    def get_mel(self, audio):
        return self.mel.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

    def inference(self, melspectrogram, iters=1000):
        x = torch.normal(0, 1e-2, size=((melspectrogram.size(-1) - 1) * MelSpectrogramConfig.hop_length, )).to(melspectrogram.device).requires_grad_()
        optimizer = torch.optim.Adam([x])
        criterion = torch.nn.MSELoss()

        def closure():
            optimizer.zero_grad()
            mel = self.get_mel(x)
            loss = criterion(mel, melspectrogram[0])
            loss.backward()
            return loss

        # with tqdm() as pbar:
        for i in range(iters):
            loss = optimizer.step(closure=closure)
                # pbar.set_postfix(loss=criterion(self.get_mel(x), melspectrogram[0]).item())
        return x.detach()[None, :]

class LibrosaVocoder():
    def inference(self, mel, n_iter=1000):
        mel_spec = torch.exp(torch.clamp(mel, max=10)).cpu().numpy()[0]
        S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=MelSpectrogramConfig.sr, n_fft=MelSpectrogramConfig.n_fft)
        y_inv = librosa.griffinlim(S_inv, n_iter=n_iter,
                                    hop_length=MelSpectrogramConfig.hop_length)
        return torch.from_numpy(y_inv[None, :])
