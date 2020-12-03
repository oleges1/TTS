import torch
from torch import nn
import torch.nn.functional as F
from wavenet.model.layers import CausalConv1d, ResBlock


class WaveNet(nn.Module):
    def __init__(self,
        n_classes=256, aux_channels=42, n_channels=256,
        skip_channels=256, fast_inference=False, dilation_depth=10,
        dilation_repeat=3, kernel_size=2, inference_strategy='argmax',
        pad_value=MelSpectrogramConfig.pad_value
    ):
        self.conv1 = CausalConv1d(n_classes, n_channels, kernel_size)
        self.layer_dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.min_time = (self.kernel_size - 1) * sum(self.dilations) + 1
        self.net = nn.ModuleList()
        for i, dilation in enumerate(self.layer_dilations):
            self.net.append(ResBlock(
                dilation, aux_channels, n_channels, skip_channels, kernel_size, fast_inference=fast_inference
            ))
        self.postnet = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(skip_channel, skip_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv1d(skip_channel, n_classes, kernel_size=1, bias=True)
        )
        self.fast_inference = fast_inference
        self.inference_strategy = inference_strategy
        self.pad_value = pad_value
        self.n_classes = n_classes

    def forward(self, x, h):
        # x - (b, n_classes, T)
        # h - (b, aux_channels, T)
        x = self.conv1(x)
        skip_sum = None
        for layer in self.net:
            x, skip = layer(x, h)
            if skip_sum is not None:
                skip_sum = skip_sum + skip
            else:
                skip_sum = skip
        return self.postnet(skip_sum)

    def generate(self, x, h):
        # x - (1, n_classes, T0)
        # h - (1, aux_channels, T)
        if self.fast_inference:
            for layer in self.net:
                layer.clear()
        n_pad = self.min_time - x.shape[1]
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), value=0.)
            h = F.pad(h, (n_pad, 0), value=self.pad_value)
        output = x
        for _ in range(h.shape[-1] - x.shape[-1]):
            x_input = output[:, :, -self.min_time:]
            h_input = h[:, :, -self.min_time + output.shape[-1]:output.shape[-1]]
            logprobs = self(x_input, h_input)

            if self.inference_strategy == "sample":
                probs = F.softmax(logprobs[0, -1], dim=0)
                sample = torch.multinomial(posterior, 1)
            elif self.inference_strategy == "argmax":
                sample = logprobs[0, -1].argmax()
            else:
                raise ValueError('Unknown inference_strategy')
            ohe_sample = F.one_hot(sample, num_classes=self.n_classes).permute(1, 0)[None] # (1, n_classes, 1)
            output = torch.cat([output, ohe_sample], dim=-1)
        return output.argmax(dim=1) # (1, n_classes, T) -> (1, T)
