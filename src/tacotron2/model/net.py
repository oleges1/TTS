import torch
from torch import nn
from math import sqrt
from tacotron2.model.blocks import CRNNEncoder, TacotronDecoder, Postnet
from tacotron2.data.transforms import MelSpectrogramConfig

class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(
            len(config.alphabet), config.encoder.in_channels)
        std = sqrt(2.0 / (len(config.alphabet) + config.encoder.in_channels))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = CRNNEncoder(**config.encoder)
        self.decoder = TacotronDecoder(**config.decoder)
        self.postnet = Postnet(**config.postnet)
        self.config = config
        self.pad_value = config.dataset.get('pad_value', MelSpectrogramConfig.pad_value)

    def forward(
            self,
            text_inputs,
            lengths=None,
            mels=None,
            output_lengths=None
        ):
        embedded_inputs = self.embedding(text_inputs)
        encoder_outputs = self.encoder(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels=mels, lengths=lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        if output_lengths is not None:
            batch_size, max_length, hidden_size = mels.shape
            mask = torch.arange(max_length, device=output_lengths.device,
                            dtype=output_lengths.dtype)[None, :] < output_lengths[:, None]

            mel_outputs.data.masked_fill_(mask[..., None], self.pad_value)
            mel_outputs_postnet.data.masked_fill_(mask[..., None], self.pad_value)
            gate_outputs.data.masked_fill_(mask, 1e3)
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
