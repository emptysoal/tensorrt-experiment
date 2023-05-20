import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image, mask_true = batch["image"], batch["mask"]

            # move images and labels to correct device and type
            image = image.to(device)
            mask_true = mask_true.to(device)

            # predict the mask
            mask_pred = net(image)

            loss = nn.CrossEntropyLoss()(mask_pred, mask_true)
            loss += dice_loss(
                F.softmax(mask_pred, dim=1).float(),
                F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            total_loss += loss.item()

            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1), total_loss / max(num_val_batches, 1)
