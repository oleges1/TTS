import torch
from torch import nn
import torch.nn.functional as F
from tacotron2.model.attention import LocationSensitiveAttention, MonotonicLocationSensitiveAttention
from tacotron2.model.utils import Linears, ZoneOutCell

class CRNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        cnn_layers=2,
        cnn_dropout=0.5,
        rnn_layers=2,
        hidden_size=None,
        kernel_size=9,
        dropout=0.2
    ):
      super(CRNNEncoder, self).__init__()
      self.kernel_size = kernel_size
      self.cnn_layers = cnn_layers

      prev_channels = in_channels
      channels = hidden_size if hidden_size is not None else in_channels
      layers = []
      for _ in range(cnn_layers):
          layers.append(nn.Conv1d(
              prev_channels, channels, kernel_size=kernel_size,
              padding=(kernel_size - 1) // 2, bias=False
          ))
          layers.append(nn.BatchNorm1d(channels))
          layers.append(nn.ReLU())
          if cnn_dropout > 0:
              layers.append(nn.Dropout(cnn_dropout))
          prev_channels = channels

      self.cnn_net = nn.Sequential(
          *layers
      ) if len(layers) > 0 else None

      self.rnn = nn.GRU(
          input_size=prev_channels,
          hidden_size=hidden_size,
          num_layers=rnn_layers,
          dropout=dropout,
          batch_first=True,
          bidirectional=True
      )


    def forward(
        self,
        input,
        last_h=None
    ):
      # input (batch_size, max_length, hidden_size)
      batch_size, max_length, _ = input.shape
      if self.cnn_net is not None:
          input = self.cnn_net(input.permute(0, 2, 1)).permute(0, 2, 1)
      output, h = self.rnn(input, last_h)
      # B, T, H
      return output


class Postnet(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        embedding_dim=512,
        kernel_size=5,
        num_layers=5,
        dropout_prob=0.5
    ):
        super(Postnet, self).__init__()
        layers = []
        in_channels = n_mel_channels
        for i in range(num_layers):
            out_channels = embedding_dim if i != num_layers - 1 else n_mel_channels
            layers.append(nn.Conv1d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            padding=int((kernel_size - 1) / 2),
                            bias=(i == num_layers - 1)))
            torch.nn.init.xavier_uniform_(
                layers[-1].weight, gain=torch.nn.init.calculate_gain('tanh'))
            layers.append(nn.BatchNorm1d(out_channels))
            if (i != num_layers - 1):
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout_prob))
            in_channels = out_channels
        self.net = nn.Sequential(
            *layers
        )


    def forward(self, x):
        # x: B, T, H
        x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x



class TacotronDecoder(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        prenet_dim=256,
        prenet_layers=3,
        attention_rnn_dim=512,
        decoder_rnn_dim=512,
        encoder_embedding_dim=512,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        teacher_forcing_ratio=1.,
        dropout_prob=0.5,
        zoneout_prob=0.1,
        gate_thr=0.5,
        use_monotonic_attention=False
    ):
        super(TacotronDecoder, self).__init__()
        self.prenet = Linears(n_mel_channels, prenet_dim,
            num_layers=prenet_layers, bias=False, activation='relu', dropout=dropout_prob)
        self.prenet_dim = prenet_dim
        self.prenet_layers = prenet_layers
        self.attention_rnn = ZoneOutCell(
            nn.LSTMCell(
                input_size=prenet_dim + encoder_embedding_dim,
                hidden_size=attention_rnn_dim
            ), zoneout_prob=zoneout_prob)
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn = ZoneOutCell(
            nn.LSTMCell(
                input_size=attention_rnn_dim + encoder_embedding_dim,
                hidden_size=decoder_rnn_dim
            ), zoneout_prob=zoneout_prob)
        self.decoder_rnn_dim = decoder_rnn_dim
        if use_monotonic_attention:
            self.attention_layer = MonotonicLocationSensitiveAttention(
                attention_rnn_dim, encoder_embedding_dim,
                attention_dim, attention_location_n_filters,
                attention_location_kernel_size
            )
        else:
            self.attention_layer = LocationSensitiveAttention(
                attention_rnn_dim, encoder_embedding_dim,
                attention_dim, attention_location_n_filters,
                attention_location_kernel_size
            )
        self.output_project = Linears(
            decoder_rnn_dim + encoder_embedding_dim, n_mel_channels)
        self.gate = Linears(
            decoder_rnn_dim + encoder_embedding_dim, 1, num_layers=1,
            bias=True, w_init_gain='sigmoid')
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout_prob = dropout_prob
        self.n_mel_channels = n_mel_channels
        self.gate_thr = gate_thr


    def init_states(self, encoder_out):
        batch_size, max_length, hidden_size = encoder_out.shape
        self.attention_hidden = torch.zeros((batch_size, self.attention_rnn_dim), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        self.attention_cell =  torch.zeros_like(self.attention_hidden)
        self.decoder_hidden = torch.zeros((batch_size, self.decoder_rnn_dim), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        self.decoder_cell =  torch.zeros_like(self.decoder_hidden)
        self.attention_context = torch.zeros((batch_size, self.encoder_embedding_dim), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        self.attention_weights = torch.zeros((batch_size, max_length), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        self.attention_weights_sum = torch.zeros((batch_size, max_length), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        self.processed_memory = self.attention_layer.memory(encoder_out)


    def append_first_frame(self, encoder_out, mels):
        batch_size, max_length, hidden_size = encoder_out.shape
        if mels is not None:
            decoder_inputs = torch.cat([
                torch.zeros((batch_size, 1, self.n_mel_channels), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True),
                mels
            ], dim=1)
        else:
            decoder_inputs = torch.zeros((batch_size, 1, self.n_mel_channels), dtype=encoder_out.dtype, device=encoder_out.device, requires_grad=True)
        return decoder_inputs

    def step(self, state, encoder_out, mask):
        attention_input = torch.cat([state, self.attention_context], dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            attention_input, (self.attention_hidden, self.attention_cell))
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_sum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, encoder_out, self.processed_memory,
            attention_weights_cat, mask)
        self.attention_weights_sum = self.attention_weights_sum + self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.output_project(
            decoder_hidden_attention_context)
        gate_prediction = self.gate(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights


    def forward(
        self,
        encoder_out,  # (batch_size, max_length, hidden_size)
        lengths=None,  # None for inference
        mels=None
    ):
        decoder_inputs = self.append_first_frame(encoder_out, mels)
        batch_size, max_length, _ = encoder_out.shape
        if lengths is not None:
            batch_size, max_length, _ = encoder_out.shape
            mask = torch.arange(max_length, device=lengths.device,
                            dtype=lengths.dtype)[None, :] < lengths[:, None]
            mask = ~(mask.bool())
        else:
            mask = None
        self.init_states(encoder_out)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            if (len(mel_outputs) == 0) or (mels is not None and torch.rand(1) < self.teacher_forcing_ratio):
                decoder_input = decoder_inputs[:, len(mel_outputs)]
            else:
                decoder_input = mel_outputs[-1][:, 0]
            decoder_input = self.prenet(decoder_input)
            predicted_mel, predicted_gate, attention_weights = self.step(
                decoder_input, encoder_out, mask)
            mel_outputs.append(predicted_mel.unsqueeze(1))
            gate_outputs.append(predicted_gate)
            alignments.append(attention_weights.unsqueeze(1))
            if (
                (mels is not None and len(mel_outputs) == mels.shape[1]) or
                (mels is None and (torch.sigmoid(predicted_gate).item() > self.gate_thr or len(mel_outputs) > 20 * max_length))
            ):
                break
        # parse outputs:
        alignments = torch.cat(alignments, dim=1)     # B, T, H
        gate_outputs = torch.cat(gate_outputs, dim=1) # B, T
        mel_outputs = torch.cat(mel_outputs, dim=1)   # B, T, H
        return mel_outputs, gate_outputs, alignments


class UncoditionalDecoder(TacotronDecoder):
    def __init__(self, *args, **kwargs):
        super(UncoditionalDecoder, self).__init__(*args, **kwargs)
        self.encoder = Linears(self.n_mel_channels, self.prenet_dim,
            num_layers=self.prenet_layers, bias=False, activation='relu', dropout=self.dropout_prob)

    def init_states(self, batch_size, max_length, dtype, device):
        self.attention_hidden = torch.zeros((batch_size, self.attention_rnn_dim), dtype=dtype, device=device)
        self.attention_cell =  torch.zeros_like(self.attention_hidden)
        self.decoder_hidden = torch.zeros((batch_size, self.decoder_rnn_dim), dtype=dtype, device=device)
        self.decoder_cell =  torch.zeros_like(self.decoder_hidden)
        self.attention_context = torch.zeros((batch_size, self.encoder_embedding_dim), dtype=dtype, device=device)
        self.attention_weights = torch.zeros((batch_size, max_length), dtype=dtype, device=device)
        self.attention_weights_sum = torch.zeros((batch_size, max_length), dtype=dtype, device=device)

    def append_first_frame(self, batch_size, mels, dtype, device):
        if mels is not None:
            decoder_inputs = torch.cat([
                torch.empty((batch_size, 1, self.n_mel_channels), dtype=dtype, device=device).uniform_(),
                mels
            ], dim=1)
        else:
            decoder_inputs = torch.empty((batch_size, 1, self.n_mel_channels), dtype=dtype, device=device).uniform_()
        return decoder_inputs


    def step(self, state, encoder_out, mask):
        max_len = encoder_out.shape[1]
        attention_input = torch.cat([state, self.attention_context], dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            attention_input, (self.attention_hidden, self.attention_cell))
        attention_weights_cat = torch.cat(
            (self.attention_weights[:, :max_len].unsqueeze(1),
             self.attention_weights_sum[:, :max_len].unsqueeze(1)), dim=1)
        self.attention_context, attention_weights = self.attention_layer(
            self.attention_hidden, encoder_out, self.attention_layer.memory(encoder_out),
            attention_weights_cat, mask[:, :max_len] if mask is not None else None)
        self.attention_weights[:, :max_len] = self.attention_weights[:, :max_len] + attention_weights
        self.attention_weights_sum = self.attention_weights_sum + self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.output_project(
            decoder_hidden_attention_context)
        return decoder_output, self.attention_weights


    def forward(
        self,
        lengths,
        mels=None # teacher forcing
    ):
        if mels is not None:
            batch_size = lengths.shape[0]
            mask = torch.arange(max_length, device=lengths.device,
                            dtype=lengths.dtype)[None, :] < lengths[:, None]
            mask = ~(mask.bool())
        else:
            mask = None
            max_length = lengths.item()
            batch_size = 1
        self.init_states(batch_size, max_length, dtype=torch.float, device=lengths.device)
        decoder_inputs = self.append_first_frame(batch_size, mels, dtype=torch.float, device=lengths.device)

        mel_outputs, alignments = [], []
        decoder_inputs_processed = []
        while True:
            if (len(mel_outputs) == 0) or (mels is not None and torch.rand(1) < self.teacher_forcing_ratio):
                decoder_input = decoder_inputs[:, len(mel_outputs)]
            else:
                decoder_input = mel_outputs[-1][:, 0]
            decoder_inputs_processed.append(self.encoder(decoder_input).unsqueeze(1))
            decoder_input = self.prenet(decoder_input)

            predicted_mel, attention_weights = self.step(
                decoder_input, torch.cat(decoder_inputs_processed, dim=1), mask)
            mel_outputs.append(predicted_mel.unsqueeze(1))
            alignments.append(attention_weights.unsqueeze(1))
            if (
                (mels is not None and len(mel_outputs) == mels.shape[1]) or
                (mels is None and (len(mel_outputs) == max_length))
            ):
                break
        # parse outputs:
        mel_outputs = torch.cat(mel_outputs, dim=1)   # B, T, H
        alignments = torch.cat(alignments, dim=1)     # B, T, H
        return mel_outputs, alignments
