import torch
from torch import nn
import torch.nn.functional as F
from tacotron2.model.attention import LocationSensitiveAttention

class Linears(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, bias=True,
                w_init_gain='linear', dropout=0, activation=None):
        super(Linears, self).__init__()
        linear_layers = []
        for i in range(num_layers):
            module = nn.Linear(in_dim, out_dim, bias=bias)
            nn.init.xavier_uniform_(
                module.weight,
                gain=nn.init.calculate_gain(w_init_gain))
            linear_layers.append(module)

            in_dim = out_dim
        self.linear_layers = nn.ModuleList(linear_layers)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if self.activation is not None:
                x = getattr(F, self.activation)(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=True)
        return x


# ref: https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/recurrent.py
class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout_prob=0):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob

    def forward(self, inputs, hidden):
        def zoneout(h, next_h, prob):
            if isinstance(h, tuple):
                num_h = len(h)
                if not isinstance(prob, tuple):
                    prob = tuple([prob] * num_h)
                return tuple([zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
            mask = h.new_tensor(h.size()).bernoulli_(prob)
            return mask * next_h + (1 - mask) * h

        next_hidden = self.cell(inputs, hidden)
        if self.training:
            next_hidden = zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden


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
                layers[-1].weight, gain=torch.nn.init.calculate_gain(w_init_gain='tanh'))
            layers.append(nn.BatchNorm1d(embedding_dim))
            if (i != num_layers - 1):
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout_prob))
            in_channels = out_channels
        self.net = nn.Sequential(
            *layers
        )


    def forward(self, x):
        x = self.net(x)
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
        zoneout_prob=0.1
    ):
        super(TacotronDecoder, self).__init__()
        self.prenet = Linears(n_mel_channels, prenet_dim,
            num_layers=prenet_layers, bias=False, activation='relu', dropout=0.5)
        self.attention_rnn = ZoneOutCell(
            nn.GRUCell(
                input_dim=prenet_dim + encoder_embedding_dim,
                hidden_size=attention_rnn_dim
            ), zoneout_prob=zoneout_prob)
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn = ZoneOutCell(
            nn.GRUCell(
                input_dim=prenet_dim + encoder_embedding_dim,
                hidden_size=decoder_rnn_dim
            ), zoneout_prob=zoneout_prob)
        self.decoder_rnn_dim = decoder_rnn_dim
        self.attention_layer = LocationSensitiveAttention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size
        )
        self.output_project = Linears(
            decoder_rnn_dim + encoder_embedding_dim, n_mel_channels)
        self.gate = Linears(
            decoder_rnn_dim + encoder_embedding_dim, num_layers=1,
            bias=True, w_init_gain='sigmoid')
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout_prob = dropout_prob
        self.n_mel_channels = n_mel_channels
        self.init_states()


    def init_states(self, encoder_out):
        batch_size, max_length, hidden_size = encoder_out.shape
        self.attention_hidden = torch.zeros((batch_size, self.attention_rnn_dim), dtype=encoder_out.dtype, device=encoder_out.device)
        self.decoder_hidden = torch.zeros((batch_size, self.decoder_rnn_dim), dtype=encoder_out.dtype, device=encoder_out.device)
        self.attention_context = torch.zeros((batch_size, self.encoder_embedding_dim), dtype=encoder_out.dtype, device=encoder_out.device)
        self.attention_weights = torch.zeros((batch_size, max_length), dtype=encoder_out.dtype, device=encoder_out.device)
        self.attention_weights_sum = torch.zeros((batch_size, max_length), dtype=encoder_out.dtype, device=encoder_out.device)
        self.processed_memory = self.attention_layer.memory(encoder_out)


    def append_first_frame(self, encoder_out):
        batch_size, max_length, hidden_size = encoder_out.shape
        decoder_inputs = torch.cat([
            torch.zeros((batch_size, 1, hidden_size), dtype=encoder_out.dtype, device=encoder_out.device),
            encoder_out
        ], dim=1)
        return decoder_inputs

    def step(self, state, encoder_out, mask):
        attention_input = torch.cat([state, self.attention_context], dim=-1)
        self.attention_hidden = self.attention_rnn(attention_input, self.attention_hidden)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_sum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, encoder_out, self.processed_memory,
            attention_weights_cat, mask)
        self.attention_weights_sum += self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden = self.decoder_rnn(decoder_input, self.decoder_hidden)

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
        if mels is not None:
            assert mels.shape == encoder_out.shape, str(mels.shape) + '- mels, encoder_out-' + str(encoder_out.shape)
            if self.teacher_forcing_ratio != 1:
                mels[:int((1 - self.teacher_forcing_ratio) * len(mels))] = encoder_out[:int((1 - self.teacher_forcing_ratio) * len(mels))]
            decoder_inputs = self.append_first_frame(mels)
        else:
            decoder_inputs = self.append_first_frame(encoder_out)

        batch_size, max_length, hidden_size = decoder_input.shape
        if lengths is not None:
            mask = torch.arange(max_length, device=lengths.device,
                            dtype=length.dtype)[None, :] < length[:, None]
        else:
            mask = None
        self.init_states(encoder_out)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            predicted_mel, predicted_gate, attention_weights = self.step(
                decoder_inputs[:, len(mel_outputs)], encoder_out, mask)
            mel_outputs.append(predicted_mel.squeeze(1))
            gate_outputs.append(predicted_gate.squeeze(1))
            alignments.append(attention_weights)
        # parse outputs:
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            batch_size, max_length, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments
