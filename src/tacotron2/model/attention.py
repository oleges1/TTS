import torch
from torch import nn
import torch.nn.functional as F
from tacotron2.model.utils import Linears

class LocationBlock(nn.Module):

    def __init__(
        self,
        attention_n_filters,
        attention_kernel_size,
        attention_dim
    ):
        super().__init__()

        padding = int((attention_kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            2, attention_n_filters, kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1, dilation=1
        )
        self.projection = Linears(attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weights):
        output = self.conv(attention_weights).transpose(1, 2)
        output = self.projection(output)
        return output



class LocationSensitiveAttention(nn.Module):

    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
        init_r=-4
    ):
        super().__init__()

        self.query = Linears(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory = Linears(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = Linears(attention_dim, 1, bias=False)
        self.location_layer = LocationBlock(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.r = nn.Parameter(torch.Tensor([init_r]))
        self.score_mask_value = -1e20

    def get_alignment_energies(
        self,
        query,
        processed_memory,
        attention_weights_cat
    ):
        """
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        """
        processed_query = self.query(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)

        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        )) + self.r

        energies = energies.squeeze(2)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask
    ):
        """
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            # fill inplace:
            alignment = alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class MonotonicLocationSensitiveAttention(LocationSensitiveAttention):
    # with help: https://github.com/j-min/MoChA-pytorch
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def gaussian_noise(self, tensor_like):
        """Additive gaussian nosie to encourage discreteness"""
        return torch.empty_like(tensor_like).normal_()

    def log_safe_cumprod(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1)

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]
        * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
        * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
        """
        batch_size, sequence_length = x.size()
        if torch.cuda.is_available():
            one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        else:
            one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask
    ):
        if attention_weights_cat.sum() == 0:
            # first step
            alpha = torch.empty_like(attention_weights_cat[:, 0], requires_grad=True)
            alpha = 0.
            alpha[:, 0] = 1.
            attention_weights = alpha
        else:
            alignment = super().get_alignment_energies(
                    attention_hidden_state, processed_memory, attention_weights_cat
            )
            if self.training:
                # soft:
                alignment = alignment + self.gaussian_noise(alignment)
                if mask is not None:
                    # fill inplace:
                    alignment = alignment.data.masked_fill_(mask, self.score_mask_value)

                p_select = self.sigmoid(alignment)
                log_cumprod_1_minus_p = self.log_safe_cumprod(1 - p_select)
                log_attention_weights_prev = torch.log(torch.clamp(attention_weights_cat[:, 0], min=1e-10))
                alpha = p_select * torch.exp(log_cumprod_1_minus_p) * torch.cumsum(torch.exp(log_attention_weights_prev - log_cumprod_1_minus_p), dim=1)

                attention_weights = alpha
            else:
                # hard:
                above_threshold = (alignment > 0).float()

                p_select = above_threshold * torch.cumsum(attention_weights_cat[:, 0], dim=1)
                attention = p_select * self.exclusive_cumprod(1 - p_select)

                # Not attended => attend at last encoder output
                # Assume that encoder outputs are not padded (this is true on inference)
                attended = attention.sum(dim=1)
                for batch_i in range(attention_weights_cat.shape[0]):
                    if not attended[batch_i]:
                        attention[batch_i, -1] = 1
                attention_weights = attention
        # apply attention:
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights
