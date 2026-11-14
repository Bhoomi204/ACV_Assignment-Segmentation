import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassSoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 21, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return logits.sum() * 0.0

        probs = F.softmax(logits.float(), dim=1)
        clean_targets = torch.where(valid, targets, torch.zeros_like(targets))
        one_hot = F.one_hot(clean_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        valid_4d = valid.unsqueeze(1).float()
        probs = probs * valid_4d
        one_hot = one_hot * valid_4d

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs + one_hot, dim=dims)
        present = (torch.sum(one_hot, dim=dims) > 0)
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice_per_class[present].mean()

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes: int = 21, ignore_index: int = 255, bnd_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = MultiClassSoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.bnd_weight = bnd_weight

    def forward(self, seg_logits, bnd_logits, targets, bnd_targets, bnd_valid):
        loss_seg = self.ce(seg_logits.float(), targets) + self.dice(seg_logits, targets)
        if bnd_logits is None:
            return loss_seg

        bnd_f = bnd_logits.squeeze(1).float()
        bce_map = F.binary_cross_entropy_with_logits(bnd_f, bnd_targets, reduction="none")
        loss_bnd_bce = (bce_map * bnd_valid).sum() / (bnd_valid.sum() + 1e-6)

        bnd_prob = torch.sigmoid(bnd_f) * bnd_valid
        bnd_gt = bnd_targets * bnd_valid
        inter = (bnd_prob * bnd_gt).sum()
        denom = bnd_prob.sum() + bnd_gt.sum()
        loss_bnd_dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        return loss_seg + self.bnd_weight * (loss_bnd_bce + loss_bnd_dice)

class ConfusionMatrixMeter:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        valid = (targets != self.ignore_index)
        p = preds[valid].view(-1).cpu()
        t = targets[valid].view(-1).cpu()
        k = (t >= 0) & (t < self.num_classes)
        inds = self.num_classes * t[k] + p[k]
        self.mat += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        tp = torch.diag(self.mat).float()
        fp = self.mat.sum(dim=0).float() - tp
        fn = self.mat.sum(dim=1).float() - tp
        denom = tp + fp + fn
        valid_cls = denom > 0
        iou_per_class = torch.where(valid_cls, tp / denom.clamp(min=1), torch.tensor(float("nan")))
        miou = iou_per_class[valid_cls].mean().item() * 100.0
        pixel_acc = (tp.sum() / self.mat.sum().clamp(min=1)).item() * 100.0
        return iou_per_class.numpy() * 100.0, miou, pixel_acc
