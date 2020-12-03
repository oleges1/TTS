import torch
from torch import nn
import torch.nn.functional as F
import collections

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super(CausalConv1d, self).__init__()
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class ResBlock(nn.Module):
    def __init__(self, dilation, aux_channels, n_channels, skip_channels, kernel_size, fast_inference=False):
        super(ResBlock, self).__init__()
        conv_dilation = 1 if fast_inference else dilation
        self.filter = CausalConv1d(n_channels, n_channels, kernel_size=kernel_size, dilation=conv_dilation)
        self.gate = CausalConv1d(n_channels, n_channels, kernel_size=kernel_size, dilation=conv_dilation)
        self.aux_filter = nn.Conv1d(aux_channels, n_channels, kernel_size=1)
        self.aux_gate = nn.Conv1d(aux_channels, n_channels, kernel_size=1)
        if fast_inference:
            self.queue = None
            self.buffer_size = conv_dilation * 2 * (kernel_size - 1)
        self.fast_inference = fast_inference
        self.permute = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.skip = nn.Conv1d(n_channels, skip_channels, kernel_size=1)

    def forward(self, x, h):
        if self.fast_inference:
            pass
            # if queue is empty - store
            # else update queue via torch.cat

        else:
            out_tanh = torch.tanh(self.filter(x) + self.aux_filter(h))
            out_gate = torch.sigmoid(self.gate(x) + self.aux_gate(h))
            output = self.permute(out_tanh * out_gate)
            skip = self.skip(output)
            return output, skip

    def clear(self):
        self.queue = None


class UpSampleMel(nn.Module):
    """UPSAMPLING LAYER WITH Conv and Billinear Upsample.
    """

    def __init__(self):
        super(UpSampling, self).__init__()

    def forward(self, h, shape):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = shape.
        """
        h = h.unsqueeze(1)  # B x 1 x C x T
        target_shape = h.size()
        target_shape[-1] = shape
        h = F.interpolate(h, size=target_shape[2:], mode='bilinear')
        return h.squeeze(1)
