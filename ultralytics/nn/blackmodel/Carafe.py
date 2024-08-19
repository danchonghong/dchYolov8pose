'''
    https://arxiv.org/abs/1905.02188
    c:输入通道数
    scale:上采样扩大尺寸倍数，h*w -> (h*scale)*(w*scale)
'''
import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale
        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, x):
        b, c, h, w = x.size()
        h_, w_ = h * self.scale, w * self.scale

        w = self.comp(x)  # b * m * h * w
        w = self.enc(w)  # b * 100 * h * w
        w = self.pix_shf(w)  # b * 25 * h_ * w_
        w = torch.softmax(w, dim=1)  # b * 25 * h_ * w_

        x = self.upsmp(x)  # b * c * h_ * w_
        x = self.unfold(x)  # b * 25c * h_ * w_
        x = x.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        x = torch.einsum('bkhw,bckhw->bchw', [w, x])  # b * c * h_ * w_
        return x


if __name__ == '__main__':
    v = CARAFE(32)
    print(v(torch.Tensor(1, 32, 40, 40)).shape)
