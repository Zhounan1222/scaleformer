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



class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn', groups=1):
        super(EnBlock, self).__init__()
        
        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups)
        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups)

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
    def __init__(self, in_channels, out_channels, groups=1):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=groups)

    def forward(self, x):
        y = self.conv(x)

        return y


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_type='interpolate'):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) if up_type == 'interpolate' else nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x, enc):
        x = self.up(x)
        return x + enc


class Unet1(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_classes=3):
        super(Unet1, self).__init__()

        self.InitConv = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)

        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels*2)
        self.skip_fuse1 = nn.Sequential(normalization(base_channels*4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(in_channels=base_channels*4, out_channels=base_channels, kernel_size=1))
        
        self.EnBlock2 = nn.Sequential(EnBlock(in_channels=base_channels*8, groups=4),
                                      EnBlock(in_channels=base_channels*8, groups=4))

        self.EnDown2 = EnDown(in_channels=base_channels*8, out_channels=base_channels*4, groups=4)
        self.skip_fuse2 = nn.Sequential(normalization(base_channels*8),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(in_channels=base_channels*8, out_channels=base_channels*2, kernel_size=1))

        self.bn = normalization(base_channels*4) 
        self.relu = nn.ReLU(inplace=True)
        self.extract_kernel_0 = nn.Conv3d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=1)
        self.extract_kernel_1 = nn.Conv3d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, padding=1)
        self.extract_kernel_2 = nn.Conv3d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=5, padding=2)

        self.fuseEnBlock3 = nn.Sequential(nn.Conv3d(in_channels=base_channels*12, out_channels=base_channels*4, kernel_size=1),
                                          EnBlock(in_channels=base_channels * 4),
                                          EnBlock(in_channels=base_channels * 4))

        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels*8)
        self.EnBlock4 = nn.Sequential(EnBlock(in_channels=base_channels * 8),
                                      EnBlock(in_channels=base_channels * 8),
                                      EnBlock(in_channels=base_channels * 8),
                                      EnBlock(in_channels=base_channels * 8))

        self.dec_up_2 = UpBlock(base_channels*8, base_channels*4, up_type='ConvTrans')
        self.dec_block_2 = EnBlock(base_channels*4)
        self.dec_up_1 = UpBlock(base_channels*4, base_channels*2, up_type='ConvTrans')
        self.dec_block_1 = EnBlock(base_channels*2)
        self.dec_up_0 = UpBlock(base_channels*2, base_channels, up_type='ConvTrans')
        self.dec_block_0 = EnBlock(base_channels)
        self.dec_end = nn.Conv3d(base_channels, num_classes, 1, 1, 0, bias=True)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.view(-1, 1, H, W, D)
        x = self.InitConv(x)       # (4N, 16, 128, 128, 128)
        x = self.dropout(x)

        # down-sampling
        # x = torch.utils.checkpoint.checkpoint(self.EnBlock1, x)
        x = self.EnBlock1(x)
        x1_1 = x

        x = self.EnDown1(x)  # (4N, 32, 64, 64, 64)
        _, C, H, W, D = x.shape
        x = x.view(B, -1, H, W, D)
        # x = torch.utils.checkpoint.checkpoint(self.EnBlock2, x)
        x = self.EnBlock2(x)
        x2_1 = x

        x = self.EnDown2(x)  # (4N, 256, 32, 32, 32)
        fea = self.relu(self.bn(x))  # (1, 128, 16, 16, 16)
        fuse_fea0 = self.extract_kernel_0(fea)
        fuse_fea1 = self.extract_kernel_1(fea)
        fuse_fea2 = self.extract_kernel_2(fea)

        x = torch.cat([fuse_fea0, fuse_fea1, fuse_fea2], dim=1)
        x = torch.utils.checkpoint.checkpoint(self.fuseEnBlock3, x)
        # x = self.fuseEnBlock3(x)
        x3_1 = x

        x = self.EnDown3(x)  # (1, 128, 16, 16, 16)
        x = torch.utils.checkpoint.checkpoint(self.EnBlock4, x)
        # x = self.EnBlock4(x)

        # up-sampling
        y = self.dec_up_2(x, x3_1)
        y = self.dec_block_2(y)  # (1, 64, 32, 32, 32)
        y = self.dec_up_1(y, self.skip_fuse2(x2_1))  #
        y = self.dec_block_1(y)
        _, C, H, W, D = x1_1.shape
        x1_1 = x1_1.view(B, -1, H, W, D)
        y = self.dec_up_0(y, self.skip_fuse1(x1_1))
        fea = self.dec_block_0(y)
        y = self.dec_end(fea)
        y = self.sigmoid(y)
        return y, fea


class Unet2(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, num_classes=3, input_size=128):
        super(Unet2, self).__init__()

        self.InitConv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.2)
        self.block1 = EnBlock(in_channels=base_channels)

        self.block2 = nn.Sequential(
            EnDown(in_channels=base_channels, out_channels=base_channels*2),
            EnBlock(in_channels=base_channels*2),
            EnBlock(in_channels=base_channels*2))

        self.block3 = nn.Sequential(
            EnDown(in_channels=base_channels*2, out_channels=base_channels*4),
            EnBlock(in_channels=base_channels * 4),
            EnBlock(in_channels=base_channels * 4))

        self.block4 = nn.Sequential(
            EnDown(in_channels=base_channels*4, out_channels=base_channels*8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8),
            EnBlock(in_channels=base_channels * 8))
        
        fc_channel = base_channels * 8 * ((input_size // 8) ** 3)
        self.linear = nn.Linear(fc_channel, 1)
        
        self.dec_up_2 = UpBlock(base_channels*8, base_channels*4, up_type='ConvTrans')
        self.dec_block_2 = EnBlock(base_channels*4)
        self.dec_up_1 = UpBlock(base_channels*4, base_channels*2, up_type='ConvTrans')
        self.dec_block_1 = EnBlock(base_channels*2)
        self.dec_up_0 = UpBlock(base_channels*2, base_channels, up_type='ConvTrans')
        self.dec_block_0 = EnBlock(base_channels)
        self.dec_end = nn.Conv3d(base_channels, num_classes, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b = x.size(0)
        x = self.InitConv(x)       # (1, 16, 128, 128, 128)
        x = self.dropout(x)        
        x1_1 = torch.utils.checkpoint.checkpoint(self.block1, x)        
        x2_1 = torch.utils.checkpoint.checkpoint(self.block2, x1_1)
        x3_1 = torch.utils.checkpoint.checkpoint(self.block3, x2_1)
        x4_4 = torch.utils.checkpoint.checkpoint(self.block4, x3_1)
        # x1_1 = self.block1(x)
        # x2_1 = self.block2(x1_1)
        # x3_1 = self.block3(x2_1)
        # x4_4 = self.block4(x3_1)
        
        # up-sampling
        y = self.dec_up_2(x4_4, x3_1)
        y = self.dec_block_2(y)
        y = self.dec_up_1(y, x2_1)
        y = self.dec_block_1(y)
        y = self.dec_up_0(y, x1_1)
        y = self.dec_block_0(y)
        y = self.dec_end(y)
        y = self.sigmoid(y)

        if not self.training:
            logit = F.interpolate(x4_4, size=(16, 16, 16))
        else:
            logit = x4_4
        et_exist = self.linear(logit.view(b, -1)) 
        return y, et_exist


class Cascaded_Unet(nn.Module):
    def __init__(self, base_channels=16, num_classes=3):
        super(Cascaded_Unet, self).__init__()
        self.stage_1 = Unet1(base_channels=16, num_classes=3)
        self.stage_2 = Unet2(base_channels=32, num_classes=3)
    
    def forward(self, x):
        pred1, fuse_fea = self.stage_1(x)
        x = torch.cat([x, fuse_fea, pred1], dim=1)
        pred2 = self.stage_2(x, fuse_fea)
        return pred1, pred2


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 120, 120, 120), device=cuda0)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = Unet2(in_channels=4, base_channels=32, num_classes=3)
        model.cuda()
        x = model(x)
        print(x.shape)
        
        # from thop import profile
        # flops, params = profile(model, (x,))
        # print('params: {}, flops: {}'.format(params / 1000000, flops / 1000000000))
