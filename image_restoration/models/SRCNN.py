import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNNBlock(nn.Module):
    def __init__(self, c, c_expand=128, kernel_sizes=[], strides=[], climgain_path=''):
        super().__init__()
        
        # self.hybridgain = HybridGain(climgain_path)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        dw_channel = c * c_expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=kernel_sizes[0], 
                                padding=0, stride=strides[0], groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=kernel_sizes[1], 
                                padding=0, stride=strides[1], groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=kernel_sizes[2], 
                        padding=0, stride=strides[2], groups=1, bias=True)


    def forward(self, x):
        x = self.CircularPadding(x, 0)
        # print(f'circular_padding: {x.shape}')
        x = self.conv1(x)
        # print(f'conv1: {x.shape}')
        x = F.relu(x)

        x = self.CircularPadding(x, 1)
        # print(f'circular_padding: {x.shape}')
        x = self.conv2(x)
        # print(f'conv2: {x.shape}')
        x = F.relu(x)

        x = self.CircularPadding(x, 2)
        # print(f'circular_padding: {x.shape}')
        x = self.conv3(x)
        # print(f'conv3: {x.shape}')

        return x
    
    def CircularPadding(self, inp, iconv):
        _, _, H, W = inp.shape
        kht, kwd = self.kernel_sizes[iconv]
        sht, swd = self.strides[iconv]
        assert kwd%2 != 0 and kht%2 !=0 and (W-kwd)%swd==0 and (H-kht)%sht ==0, 'kernel_size should be odd, (dim-kernel_size) should be divisible by stride'

        pwd = int((W - 1 - (W - kwd) / swd) // 2)
        pht = int((H - 1 - (H - kht) / sht) // 2)
        
        # kht1, kwd1 = self.kernel_sizes[1]
        # kht2, kwd2 = self.kernel_sizes[2]
        # pwd = int((W - 1 - (W - kwd) / swd) // 2 + (W - 1 - (W - kwd1) / swd) // 2 + (W - 1 - (W - kwd2) / swd) // 2)
        # pht = int((H - 1 - (H - kht) / sht) // 2 + (H - 1 - (H - kht1) / sht) // 2 + (H - 1 - (H - kht2) / sht) // 2)
        
        x = F.pad(inp, (pwd, pwd, pht, pht), 'circular')

        return x
    
class SRCNN(nn.Module):

    def __init__(self, img_channel=1, c_expand=64, kernel_sizes=[], strides=[], climgain_path=''):
        super().__init__()

        self.blk = SRCNNBlock(img_channel, c_expand, kernel_sizes, strides, climgain_path)

    def forward(self, inp):
        return self.blk(inp)
    
if __name__ == '__main__':
    model = SRCNN(img_channel=1, c_expand=64, kernel_sizes=[(27, 27), (15, 15), (9, 9)], strides=[(1, 1), (1, 1), (1, 1)])
    x = torch.rand((1, 1, 960, 240))
    x = model(x)
    

