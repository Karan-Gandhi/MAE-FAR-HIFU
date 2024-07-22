import argparse
import os
import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from ACR_model import ACR
from MAE.util import misc
from discriminator import NLayerDiscriminator
from losses import Losses

class HIFUDataset(Dataset):
    def __init__(self, img_dir, mask_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        self.image_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        
        input_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        
        input_image = Image.open(input_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        gt_image = Image.open(gt_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
            mask_image = self.transform(mask_image)
            gt_image = self.transform(gt_image)
            
        mask_image = (mask_image > 0).float()
            
        return {'img': input_image, 'mask': mask_image, 'gt': gt_image}

def get_lr_milestone_decay_with_warmup(optimizer, num_warmup_steps, milestone_steps, gamma, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_weight = 1.0
            for ms in milestone_steps:
                if ms < current_step:
                    lr_weight *= gamma
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mae_model", type=str, default="D:\IITGN\Programming\Streak-Removal-in-HIFU-Images\checkpoints\places2_wo_norm_pix.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--g_lr", type=float, default=1.0e-3)
    parser.add_argument("--d_lr", type=float, default=1.0e-4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Preparing models")
    
    mae = misc.get_mae_model(args.mae_model, mask_decoder=True).to(device)
    mae.requires_grad_(False)
    mae.mask_decoder = True
    
    acr = ACR().to(device)
    discriminator = NLayerDiscriminator().to(device)
    
    g_optimizer = torch.optim.Adam(acr.parameters(), lr=args.g_lr, betas=(0.9, 0.999), eps=1e-8)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.9, 0.999), eps=1e-8)

    g_sche = get_lr_milestone_decay_with_warmup(g_optimizer, num_warmup_steps=args.warmup, milestone_steps=[600000, 700000, 800000], gamma=0.5)
    d_sche = get_lr_milestone_decay_with_warmup(d_optimizer, num_warmup_steps=args.warmup, milestone_steps=[600000, 700000, 800000], gamma=0.5)
    
    print("Preparing dataloaders")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = HIFUDataset(args.input_dir, args.mask_dir, args.gt_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    print("Training ACR")

    losses = Losses()
    mae.eval()
    acr.train()
    
    for epoch in range(args.epochs):
        current_losses = []
        for items in tqdm(dataloader):
            items['img'] = items['img'].to(device)
            items['mask'] = items['mask'].to(device)
            items['gt'] = items['gt'].to(device)
            
            with torch.no_grad():
                mae_feats, scores = mae.forward_return_feature(items['img'], items['mask'])
                mae_feats = mae_feats.detach()
                scores = scores.detach()
            
            if epoch % 2 == 0:
                # Generator
                discriminator.requires_grad_(False)
                acr.requires_grad_(True)
                
                pred = acr.forward(items['img'], items['mask'], mae_feats, scores)
                loss = losses.getLoss(pred, items['gt'], items['mask'])

                g_optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()
                
                current_losses.append(loss.item())
            else:
                # Discriminator
                discriminator.requires_grad_(True)
                acr.requires_grad_(False)

                pred = acr.forward(items['img'], items['mask'], mae_feats, scores)                
                d_loss = losses.getDiscriminatorLoss(pred, items['img'], items['mask'])

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
        
        if epoch % 2 == 0:
            g_sche.step()
        else:
            d_sche.step()
            
        print(f"Epoch: {epoch}, Losses: {sum(current_losses) / len(current_losses)}, lr: {g_sche.get_lr()}")