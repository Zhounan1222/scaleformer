import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2, norm='gn'):
        super(InitConv, self).__init__()

        # self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        # self.bn1 = normalization(32, norm)
        # self.relu1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv3d(32, 16, kernel_size=5, padding=2)
        # self.bn2 = normalization(16, norm)
        # self.relu2 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv3d(16, 16, kernel_size=7, padding=3)
        # self.bn3 = normalization(16, norm)
        # self.relu3 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv3d(16*4, 10*4, kernel_size=3, padding=1, groups=4)
        # self.bn4 = normalization(10*4, norm)
        # self.relu4 = nn.ReLU(inplace=True)

        # self.conv5 = nn.Conv3d(10*4, 8*4, kernel_size=5, padding=2, groups=4 )
        # self.bn5 = normalization(8*4, norm)
        # self.relu5 = nn.ReLU(inplace=True)

        # self.conv6 = nn.Conv3d(8*4, 8*4, kernel_size=3, padding=1)
        # self.bn6 = normalization(8*4, norm)
        # self.relu6 = nn.ReLU(inplace=True)

        self.end_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.end_conv = nn.Conv3d(8*4, out_channels, kernel_size=1)
        # self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        # B, C, H, W, T = x.shape
        # x = x.view(-1, 1, H, W, T)

        # y = self.conv1(x)
        # y = self.relu1(self.bn1(y))
        # y = self.conv2(y) 
        # y = self.relu2(self.bn2(y))
        # y = self.conv3(y)
        # y = self.relu3(self.bn3(y))

        # y = y.view(B, -1, H, W, T)
        # y = self.conv4(y)
        # y = self.relu4(self.bn4(y))
        # y = self.conv5(y)
        # y = self.relu5(self.bn5(y))

        # y_shortcut = y
        # y = self.conv6(y)
        # y = self.relu6(self.bn6(y))
        # y = y + y_shortcut

        y = self.end_conv(x)
        # y = self.dropout(y)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

    def forward(self, x, enc):
        x = self.up(x)
        return x + enc


class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super(Unet, self).__init__()

        # self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.InitConv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)

        self.block2 = nn.Sequential(
            EnDown(in_channels=base_channels, out_channels=base_channels*2),
            EnBlock(in_channels=base_channels*2),
            EnBlock(in_channels=base_channels*2))
        

        # self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        # self.EnBlock2_1 = EnBlock(in_channels=base_channels*2)
        # self.EnBlock2_2 = EnBlock(in_channels=base_channels*2)
        # self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels*4)

        self.block3 = nn.Sequential(
            EnDown(in_channels=base_channels*2, out_channels=base_channels*4),
            EnBlock(in_channels=base_channels * 4),
            EnBlock(in_channels=base_channels * 4))

        # self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        # self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        # self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)


        self.block4 = nn.Sequential(
            EnDown(in_channels=base_channels*4, out_channels=base_channels*8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8))


            # self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
            # self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
            # self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
            # self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)



        self.dec_up_2 = UpBlock(base_channels*8, base_channels*4)
        self.dec_block_2 = EnBlock(base_channels*4)
        self.dec_up_1 = UpBlock(base_channels*4, base_channels*2)
        self.dec_block_1 = EnBlock(base_channels*2)
        self.dec_up_0 = UpBlock(base_channels*2, base_channels)
        self.dec_block_0 = EnBlock(base_channels)
        self.dec_end = nn.Conv3d(base_channels, 4, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.InitConv(x)       # (1, 16, 128, 128, 128)
        x = self.dropout(x)        
        x1_1 = torch.utils.checkpoint(self.EnBlock1, x)        
        x2_1 = torch.utils.checkpoint(self.block2, x1_1)
        x3_1 = torch.utils.checkpoint(self.block3, x2_1)
        x4_4 = torch.utils.checkpoint(self.block4, x3_1)
        
        # up-sampling
        y = self.dec_up_2(x4_4, x3_1)
        y = self.dec_block_2(y)
        y = self.dec_up_1(y, x2_1)
        y = self.dec_block_1(y)
        y = self.dec_up_0(y, x1_1)
        y = self.dec_block_0(y)
        y = self.dec_end(y)
        y = self.sigmoid(y)
        return y


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet(in_channels=4, base_channels=16, num_classes=4)
        model.cuda()
        output = model(x)
        print('output:', output.shape)
