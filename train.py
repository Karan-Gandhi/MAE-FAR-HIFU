import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from ACR_model import ACR
from MAE.util import misc
from discriminator import NLayerDiscriminator
from losses import Losses

import logging
from datetime import datetime

def setup_logger(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger
    logger = logging.getLogger('simple_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


class HIFUDataset(Dataset):
    def __init__(self, img_dir, mask_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        self.input_image_names = ([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_image_names = ([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.gt_image_names = ([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.input_image_names)
    
    def __getitem__(self, index):
        input_img_name = self.input_image_names[index]
        mask_img_name = self.mask_image_names[index]
        gt_img_name = self.gt_image_names[index // 18]
        
        input_path = os.path.join(self.img_dir, input_img_name)
        mask_path = os.path.join(self.mask_dir, mask_img_name)
        gt_path = os.path.join(self.gt_dir, gt_img_name)
        
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

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mae_model", type=str, default="/scratch/shreyans.jain/hifu/MAE-FAR/places2_wo_norm_pix.pth")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--input_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/streaked_images")
    parser.add_argument("--mask_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/streak_masks")
    parser.add_argument("--gt_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/images_without_streaks")
    parser.add_argument("--g_lr", type=float, default=1.0e-3)
    parser.add_argument("--d_lr", type=float, default=1.0e-4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Preparing models")
    
    mae = misc.get_mae_model('mae_vit_base_patch16', mask_decoder=True).to(device)
    checkpoint = torch.load(args.mae_model, map_location='cpu')
    msg = mae.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    
    mae.requires_grad_(False)
    mae.mask_decoder = True
    
    acr = ACR().to(device)
    discriminator = NLayerDiscriminator().to(device)
    
    g_optimizer = torch.optim.Adam(acr.parameters(), lr=args.g_lr, betas=(0.9, 0.999), eps=1e-8)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.9, 0.999), eps=1e-8)

    g_sche = get_lr_milestone_decay_with_warmup(g_optimizer, num_warmup_steps=args.warmup, milestone_steps=[50, 100, 150], gamma=0.5)
    d_sche = get_lr_milestone_decay_with_warmup(d_optimizer, num_warmup_steps=args.warmup, milestone_steps=[50, 100, 150], gamma=0.5)
    
    print("Preparing dataloaders")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = HIFUDataset(args.input_dir, args.mask_dir, args.gt_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    print("Training ACR")

    losses = Losses(discriminator, device)
    mae.eval()
    acr.train()
    
    time = str(datetime.now())
    logs = setup_logger("./logs/" + time + "/training.log")
    os.makedirs("./logs/" + time + "/ckpts")
    
    for epoch in range(args.epochs):
        current_losses = []
        l1 = Averager()
        l2 = Averager()
        l3 = Averager()
        l4 = Averager()
        l5 = Averager()
        
        for items in tqdm(dataloader):
            items['img'] = items['img'].to(device)
            items['mask'] = items['mask'].to(device)
            items['gt'] = items['gt'].to(device)
            
            with torch.no_grad():
                mae_feats, scores = mae.forward_return_feature(items['img'], items['mask'])
                mae_feats = mae_feats.detach()
                scores = scores.detach()
            
            # Generator
            discriminator.requires_grad_(False)
            acr.requires_grad_(True)
            
            pred = acr.forward(items['img'], items['mask'], mae_feats, scores)
            loss, other = losses.getLoss(pred, items['gt'], items['mask'])

            l1.add(other[0].item())
            l2.add(other[1].item())
            l3.add(other[2].item())
            l4.add(other[3].item())

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()
            
            current_losses.append(loss.item())
            # Discriminator
            discriminator.requires_grad_(True)
            acr.requires_grad_(False)

            pred = acr.forward(items['img'], items['mask'], mae_feats, scores)
            d_loss = losses.getDiscriminatorLoss(pred, items['img'], items['mask'])

            l5.add(d_loss.item())

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
    
        g_sche.step()
        d_sche.step()
            
        print(f"Epoch: {epoch}, Losses: {sum(current_losses) / len(current_losses)}, lr: {g_sche.get_lr()}")
        logs.info(f"Epoch: {epoch}, Losses: {sum(current_losses) / len(current_losses)}, lr: {g_sche.get_lr()}")
        logs.info(f"L1: {l1.item()}, L2: {l2.item()}, L3: {l3.item()}, L4: {l4.item()}, d_loss: {l5.item()}")
        
        torch.save(acr.state_dict(), f"./logs/{time}/ckpts/model_latest.pth")
        if epoch % 50 == 0:
            torch.save(acr.state_dict(), f"./logs/{time}/ckpts/model_{epoch}.pth")