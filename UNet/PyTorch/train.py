import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate
from unet.unet_model import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

classes_num = 32


def train_once(model, device, train_loader, criterion, optimizer, grad_scaler, epoch):
    model.train()
    step_total = len(train_loader)
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        images, masks_true = batch["image"], batch["mask"]

        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device)
        masks_true = masks_true.to(device)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
            masks_pred = model(images)
            loss = criterion(masks_pred, masks_true)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(masks_true, model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if step % 10 == 0:
            logging.info("Train Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f}, Lr:{:.5f}".format(
                epoch, args.epochs, step + 1, step_total, loss.item(), optimizer.param_groups[0]["lr"]
            ))

        epoch_loss += loss.item()

    train_loss = epoch_loss / step_total
    logging.info("Train set: Average Loss:{:.4f}".format(train_loss))

    return train_loss


def run():
    # 获取数据集目录
    data_root = args.data_dir

    # 创建dataset
    train_dataset = BasicDataset(
        images_dir=os.path.join(data_root, "images/train"),
        mask_dir=os.path.join(data_root, "labels/train"),
        scale=args.scale
    )
    val_dataset = BasicDataset(
        images_dir=os.path.join(data_root, "images/val"),
        mask_dir=os.path.join(data_root, "labels/val"),
        scale=args.scale
    )

    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    # 创建dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')
    # 构建网络
    model = UNet(n_channels=3, n_classes=classes_num, bilinear=args.bilinear)
    if args.load:
        state_dict = torch.load(args.load, map_location="cpu")
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    logging.info("Start training.")
    best_score = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_once(model, device, train_loader, criterion, optimizer, grad_scaler, epoch)
        val_score, val_loss = evaluate(model, val_loader, device, args.amp)
        scheduler.step()

        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Score", val_score, epoch)

        # save model
        if args.save_interval != -1 and epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_save_dir, "UNet_epoch%s.pth" % epoch)
            torch.save(model.state_dict(), save_path)

        if val_score > best_score:
            best_save_path = os.path.join(args.model_save_dir, "best.pth")
            torch.save(model.state_dict(), best_save_path)
            best_score = val_score
            logging.info("Succeeded saving best.pth, and val score is %.4f" % best_score)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch UNet')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument("--weight-decay", type=float, default=0.0008)
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument("--data-dir", type=str, default="../../Camvid_segment_dataset/")
    parser.add_argument("--model-save-dir", type=str, default="./model/")
    parser.add_argument("--save-interval", type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    print(args)

    writer = SummaryWriter(args.model_save_dir)

    run()
