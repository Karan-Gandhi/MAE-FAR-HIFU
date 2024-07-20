import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from ffc import FFCResnetBlock

class GatedConvolutions(nn.Module):
    def __init__(self, embeding_dim=512):
        super().__init__()
        self.embeding_dim = embeding_dim
        
        self.act = nn.ReLU(True)
        
        # 16, 16
        self.conv0 = nn.ConvTranspose2d(self.embeding_dim + 2, self.embeding_dim, kernel_size=4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(self.embeding_dim)
        self.alpha0 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        
        # 32, 32
        self.conv1 = nn.ConvTranspose2d(self.embeding_dim, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        
        # 64, 64
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        
        # 128, 128
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        # 256, 256
    
    def make_coord(self, shape, ranges=None, flatten=True):
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def implicit_upsample(self, feat, H, W):
        # feat = [b, d, 16, 16]
        [B, _, _, _] = feat.shape
        feat_coord = self.make_coord([H, W], flatten=False).to(feat.device).permute(2, 0, 1)
        feat_coord = feat_coord.unsqueeze(0).expand(B, 2, H, W).to(feat.dtype)
        print(feat_coord.shape)
        feat = torch.cat([feat, feat_coord], dim=1)
        return feat
    
    def forward(self, mae_feats):
        x = self.implicit_upsample(mae_feats, mae_feats.shape[2], mae_feats.shape[2])
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act(x)
        
        res = [x * self.alpha0]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        res.append(x * self.alpha1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        res.append(x * self.alpha2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        
        res.append(x * self.alpha3)
        
        return res[::-1]
    
def extract_patches(x, kernel=3, padding=None, stride=1):
    if padding is None:
        padding = 1 if kernel > 1 else 0
    if padding > 0:
        x = nn.ReplicationPad2d(padding)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches

class GroupConvAttention(nn.Module):
    def __init__(self):
        super(GroupConvAttention, self).__init__()

    def forward(self, x, scores, alpha):
        # get shapes [B,C,H,W]
        [batch, channel, height, width] = list(x.size())

        [_, _, hw] = scores.shape
        hs = int(np.sqrt(hw))
        ws = int(np.sqrt(hw))
        rate = int(height / hs)

        # value for back features
        vksize = int(rate * 2)  # must be rate*2 for transposeconv
        vpadding = rate // 2
        value = extract_patches(x, kernel=vksize, padding=vpadding, stride=rate)
        value = value.contiguous().reshape(batch, hs * ws, channel, vksize, vksize)  # [B,HW,C,K,K]

        # groupconv for attention (qk)v: B*[C,H,W]·B*[HW,C,K,K]->B*[C,H,W]
        scores = scores.permute(0, 2, 1)  # [B,HW,HW(softmax)]->[B,HW(softmax),HW]
        # [1,B*C,H,W]->[1,B*HW,H,W]->[B,HW,H,W]
        scores_ = scores.reshape(1, batch * hs * ws, hs, ws)  # [1,B*HW,H,W]
        value = value.reshape(batch * hs * ws, channel, vksize, vksize)  # [B*HW,C,K,K]
        y = F.conv_transpose2d(scores_, value, stride=rate, padding=vpadding, groups=batch) / 4.

        y = y.contiguous().reshape(batch, channel, height, width)  # [B,C,H,W]

        return x + y * alpha
    
class ACR(nn.Module):
    def __init__(self, mae_embed_dim=512):
        super().__init__()
        self.gated_conv = GatedConvolutions(mae_embed_dim)
        
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=mae_embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(mae_embed_dim)

        self.attn = GroupConvAttention()
        self.attn_wt1 = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        self.ffc_blocks = nn.Sequential(*[FFCResnetBlock(mae_embed_dim, 1) for _ in range(9)])
        
        self.attn_wt2 = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        
        self.convt1 = nn.ConvTranspose2d(mae_embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(256)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(128)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(64)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Sigmoid()
        
        
    def forward(self, x, mask, mae_feats, attention_scores):
        priors = self.gated_conv(mae_feats)
        x = torch.cat((x * (1 - mask), mask), dim=1)
        
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 256, 256
        x = self.conv2(x + priors[0])
        x = self.bn2(x)
        x = self.act(x)

        # 128, 128
        x = self.conv3(x + priors[1])
        x = self.bn3(x)
        x = self.act(x)

        # 64, 64
        x = self.conv4(x + priors[2])
        x = self.bn4(x)
        x = self.act(x)
        
        # 32, 32
        
        x = self.attn(x, attention_scores, self.attn_wt1)
        
        x = self.ffc_blocks(x + priors[3])
        
        x = self.attn(x, attention_scores, self.attn_wt2)
        
        x = self.convt1(x)
        x = self.bnt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        
        return x