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

import matplotlib.pyplot as plt

def save_image_grid(input_image, mask, ground_truth, prediction, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Flatten the 2D array of axes for easier indexing
    axs = axs.ravel()
    
    images = [input_image, mask, ground_truth, prediction]
    titles = ['Input Image', 'Mask', 'Ground Truth', 'Prediction']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert tensor to numpy array and move channel dimension to the end
        if torch.is_tensor(img):
            img = img.cpu().numpy()
            if img.ndim == 3:
                img = img.transpose(1, 2, 0)
        
        # Handle different number of channels
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(img)
        
        axs[i].set_title(title)
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
        
        self.input_image_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_image_names = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.gt_image_names = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

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
    parser.add_argument("--acr_model", type=str, default="/scratch/shreyans.jain/hifu/MAE-FAR-HIFU/logs/2024-07-26 14:08:35.099030/ckpts/model_latest.pth")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--input_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/streaked_images")
    parser.add_argument("--mask_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/streak_masks")
    parser.add_argument("--gt_dir", type=str, default="/scratch/shreyans.jain/hifu/Streak-Removal-in-HIFU-Images/images_without_streaks")
    parser.add_argument("--save_dir", type=str, default="/scratch/shreyans.jain/hifu/MAE-FAR-HIFU/preds")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Preparing models")
    
    mae = misc.get_mae_model('mae_vit_base_patch16', mask_decoder=True).to(device)
    checkpoint = torch.load(args.mae_model, map_location='cpu')
    msg = mae.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
        
    acr = ACR().to(device)
    checkpoint = torch.load(args.acr_model, map_location='cpu')
    msg = acr.load_state_dict(checkpoint, strict=False)
    print(msg)

    mae.requires_grad_(False)
    acr.requires_grad_(False)
    
    print("Preparing dataloaders")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = HIFUDataset(args.input_dir, args.mask_dir, args.gt_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    mae.eval()
    acr.eval()
    img_no = 0
    with torch.no_grad():
        for items in tqdm(dataloader):
            items['img'] = items['img'].to(device)
            items['mask'] = items['mask'].to(device)
            # items['gt'] = items['gt'].to(device)
            
            mae_feats, scores = mae.forward_return_feature(items['img'], items['mask'])
            mae_feats = mae_feats.detach()
            scores = scores.detach()
            
            pred = acr(items['img'], items['mask'], mae_feats, scores)
            
            for i in range(args.batch_size):
                save_image_grid(items['img'][i], items['mask'][i], items['gt'][i], pred[i], os.path.join(args.save_dir, f"{img_no}.png"))
                img_no += 1