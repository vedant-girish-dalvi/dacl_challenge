import torch
import numpy as np

NUM_CLASSES = 19

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def iou_score(preds, targets, num_classes=NUM_CLASSES):
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = torch.logical_and(pred_cls, target_cls).sum().item()
        union = torch.logical_or(pred_cls, target_cls).sum().item()
        iou = intersection / union if union != 0 else 0
        iou_per_class.append(iou)
    return np.mean(iou_per_class)