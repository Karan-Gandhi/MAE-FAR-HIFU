import torch
import torch.nn.functional as F

from pcp import ResNetPL

def feature_matching_loss(fake_features, target_features, mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res

def interpolate_mask(mask, shape, allow_scale_mask=False, mask_scale_mode='nearest'):
    assert mask is not None
    assert allow_scale_mask or shape == mask.shape[-2:]
    if shape != mask.shape[-2:] and allow_scale_mask:
        if mask_scale_mode == 'maxpool':
            mask = F.adaptive_max_pool2d(mask, shape)
        else:
            mask = F.interpolate(mask, size=shape, mode=mask_scale_mode)
    return mask

def generator_loss(discr_fake_pred: torch.Tensor, mask=None, args=None):
    fake_loss = F.softplus(-discr_fake_pred)
    # == if masked region should be treated differently
    # if (args['mask_as_fake_target'] and args['extra_mask_weight_for_gen'] > 0) or not args['use_unmasked_for_gen']:
    #     mask = interpolate_mask(mask, discr_fake_pred.shape[-2:], args['allow_scale_mask'], args['mask_scale_mode'])
    #     if not args['use_unmasked_for_gen']:
    #         fake_loss = fake_loss * mask
    #     else:
    #         pixel_weights = 1 + mask * args['extra_mask_weight_for_gen']
    #         fake_loss = fake_loss * pixel_weights

    return fake_loss.mean()


class Losses:
    def __init__(self, discriminator):
        self.resnet = ResNetPL()
        self.discriminator = discriminator
        self.weights = {'l1': 10, 'perceptual': 30, 'adverserial': 10, 'featureMatching': 100}
        
    def perceptualLoss(self, y_hat, y):
        return self.resnet(y_hat, y)
    
    def featureMatchingLoss(self, y_hat, y):
        _, real_feats = self.discriminator(y)
        _, gen_feats = self.discriminator(y_hat)
        
        return feature_matching_loss(gen_feats, real_feats)
    
    def adverserialLoss(self, y_hat, mask):
        gen_logits, _ = self.discriminator(y_hat)
        return generator_loss(discr_fake_pred=gen_logits, mask=mask)
        
    
    def l1Loss(self, y_hat, y, mask, weight_missing=0, weight_known=1):
        per_pixel_l1 = F.l1_loss(y_hat, y, reduction='none')
        l1_mask = mask * weight_missing + (1 - mask) * weight_known
        return (per_pixel_l1 * l1_mask).mean()
        
    def getLoss(self, y_hat, y, mask):
        return self.weights['l1'] * self.l1Loss(y_hat, y, mask) + \
               self.weights['perceptual'] * self.perceptualLoss(y_hat, y) + \
               self.weights['adverserial'] * self.adverserialLoss(y_hat, mask) + \
               self.weights['featureMatching'] * self.featureMatchingLoss(y_hat, y)