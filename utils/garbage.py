import math
import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv
from torch import optim

import logging
import wandb
# from evaluate import evaluate
# from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.evaluate import evaluate
from utils.data_loading import WoundDataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from unet import UNet

import argparse

# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(UNet, self).__init__()
#
#         # Max pooling for downsampling
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Contracting path (Encoder)
#         self.enc1 = self.conv_block(in_channels, 64)
#         self.enc2 = self.conv_block(64, 128)
#         self.enc3 = self.conv_block(128, 256)
#         self.enc4 = self.conv_block(256, 512)
#         self.enc5 = self.conv_block(512, 1024)
#
#         # Bottleneck
#         self.bottleneck = self.conv_block(1024, 1024)
#
#         # Expanding path (Decoder)
#         self.up5 = self.upconv_block(1024, 512)
#         self.up4 = self.upconv_block(512, 256)
#         self.up3 = self.upconv_block(256, 128)
#         self.up2 = self.upconv_block(128, 64)
#         self.up1 = self.upconv_block(64, out_channels, final_layer=True)
#
#     def conv_block(self, in_channels, out_channels):
#         """Convolutional block with two convolutions and ReLU activations."""
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def upconv_block(self, in_channels, out_channels, final_layer=False):
#         """Upsampling block."""
#         if final_layer:
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#                 nn.Sigmoid()  # or Softmax, depending on your final output type
#             )
#         else:
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#                 nn.ReLU(inplace=True)
#             )
#
#     def forward(self, x):
#         # Contracting path
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))
#         enc5 = self.enc5(self.pool(enc4))
#
#         # Bottleneck
#         bottleneck = self.bottleneck(enc5)
#
#         # Expanding path
#         up5 = self.up5(bottleneck)
#         up4 = self.up4(up5)
#         up3 = self.up3(up4)
#         up2 = self.up2(up3)
#         up1 = self.up1(up2)
#
#         return up1


# Define transforms for preprocessing

transform = transforms.Compose([
    transforms.ToTensor(),         # Convert to tensor
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
# Initialize Dataset
root_dir = "E:/data/"
dataset = WoundDataset(root_dir, transform=transform)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

dir_checkpoint = Path('./checkpoints/')



def find_pos():

    root_dir = "E:/data/"
    cropped_dir = os.path.join(root_dir, "Davinci_Processed_Copy")
    uncropped_dir = os.path.join(root_dir, "DB-749NG_UCSC")

    # Collect all cropped and uncropped file paths
    cropped_paths = glob(os.path.join(cropped_dir, "**/*.png"), recursive=True)
    uncropped_paths = glob(os.path.join(uncropped_dir, "**/*.JPG"), recursive=True)

    # Create a dictionary for cropped paths for quick lookup
    cropped_dict = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in cropped_paths
    }

    # Filter uncropped paths that have a matching cropped image
    data_pairs = [
        (path, cropped_dict.get(os.path.splitext(os.path.basename(path))[0]))
        for path in uncropped_paths
        if os.path.splitext(os.path.basename(path))[0] in cropped_dict
    ]

    for iidx in range(len(data_pairs)):
        uncropped_path, cropped_path = data_pairs[iidx]
        new_name = cropped_path.split('.')
        new_name = new_name[-2] + "_un" + ".JPG"
        if os.path.exists(new_name):
            cropped_image_new = cv.imread(new_name)
            new_name = cropped_path.split('.')
            new_name = new_name[-2] + ".JPG"
            # if not os.path.exists(new_name):
            cropped_image_new = Image.fromarray(cv.cvtColor(cropped_image_new, cv.COLOR_BGR2RGB))
            cropped_image_new.save(new_name)
            print("{} Exists, return".format(iidx))
            continue
            # return

        threshold = 0.8
        # All the 6 methods for comparison in a list
        methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
                   cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]

        try:
            method = methods[4]

            t1 = (210, 1300)
            ##Read Main and Needle Image
            imageMainRGB = cv.imread(uncropped_path)#[1200:3600, 2400:4800]
            imageNeedleRGB_Org = cv.imread(cropped_path)
            imageNeedleRGB = cv.imread(cropped_path)[t1[0]:t1[1], t1[0]:t1[1]]

            ##Split Both into each R, G, B Channel
            imageMainR, imageMainG, imageMainB = cv.split(imageMainRGB)
            imageNeedleR, imageNeedleG, imageNeedleB = cv.split(imageNeedleRGB)

            ##Matching each channel
            resultB = cv.matchTemplate(imageMainR, imageNeedleR, method)
            resultG = cv.matchTemplate(imageMainG, imageNeedleG, method)
            resultR = cv.matchTemplate(imageMainB, imageNeedleB, method)

            w, h = imageNeedleB.shape[::-1]
            ##Add together to get the total score
            res = 0.5 * resultB + 0.1 * resultG + 0.4 * resultR
            loc = np.where(res >= 10 * threshold)
            # res = cv.normalize( res, res, 0, 1, cv.NORM_MINMAX, -1)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            imageMainRGB = cv.rectangle(imageMainRGB, top_left, bottom_right, (255, 0, 0), thickness=2)

            cropped_image_new = np.zeros(np.array(imageMainRGB).shape, dtype=np.uint8)
            # cropped_image_new[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = np.array(imageNeedleRGB)
            cropped_image_new[top_left[1]-t1[0]:bottom_right[1]+(1518 - t1[1]), top_left[0]-t1[0]:bottom_right[0]+(1518 - t1[1])] = np.array(imageNeedleRGB_Org)
            cropped_image_new = Image.fromarray(cv.cvtColor(cropped_image_new, cv.COLOR_BGR2RGB))

            cropped_image_new.save(new_name)
            print("{}/{} Finished".format(iidx, len(data_pairs)))
        except:
            print("{}/{} Failed".format(iidx, len(data_pairs)))

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,):
    root_dir = "E:/data/"
    dataset = WoundDataset(root_dir, transform=transform)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long).squeeze(1)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

