import torch
from torch import nn
import torch.nn.functional as F

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