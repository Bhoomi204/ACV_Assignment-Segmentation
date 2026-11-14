import os, csv, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import VOC2012SegmentationDataset, VOC_CLASSES
from src.models import BaselineUNet, BoundaryGatedUNet
from src.losses_metrics import ConfusionMatrixMeter

def measure_latency(model, device, batch_size=1, img_size=256, warmup=50, runs=300):
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_batch = elapsed / runs
    return ms_per_batch, ms_per_batch / batch_size

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    meter = ConfusionMatrixMeter(num_classes=21, ignore_index=255)
    biou_inter, biou_union = 0.0, 0.0
    sample_records = []

    for imgs, masks, bnd_gt, bnd_valid, sample_ids in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        bnd_gt, bnd_valid = bnd_gt.to(device), bnd_valid.to(device)

        seg_logits, bnd_logits = model(imgs)
        preds = torch.argmax(seg_logits, dim=1)
        meter.update(preds, masks)

        trimap = (F.max_pool2d(bnd_gt.unsqueeze(1), 3, 1, 1).squeeze(1) > 0) & (bnd_valid > 0)
        if trimap.sum() > 0:
            correct_bnd = (preds[trimap] == masks[trimap]).float().sum().item()
            total_bnd = trimap.float().sum().item()
            biou_inter += correct_bnd
            biou_union += (2 * total_bnd - correct_bnd)

        for i in range(imgs.size(0)):
            valid = (masks[i] != 255)
            fg = valid & (masks[i] > 0)
            if fg.sum() > 0:
                inter = ((preds[i] == masks[i]) & fg).sum().item()
                union = (((preds[i] > 0) | (masks[i] > 0)) & valid).sum().item()
                img_iou = inter / max(union, 1)
                sample_records.append((img_iou, sample_ids[i], imgs[i].cpu(), masks[i].cpu(),
                                       preds[i].cpu(), None if bnd_logits is None else torch.sigmoid(bnd_logits[i, 0]).cpu()))

    per_cls_iou, miou, pixel_acc = meter.compute()
    biou = (biou_inter / max(biou_union, 1e-6)) * 100.0
    sample_records.sort(key=lambda x: x[0])
    return per_cls_iou, miou, pixel_acc, biou, sample_records

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for arch, label, color in [("baseline", "Baseline U-Net (Slide 23)", "#1f77b4"),
                               ("boundary", "Boundary-Gated U-Net (+Gated-SCNN)", "#d62728")]:
        epochs, t_loss, v_miou = [], [], []
        with open(f"logs/{arch}_train_log.csv") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                t_loss.append(float(row["train_loss"]))
                v_miou.append(float(row["val_miou"]))
        axes[0].plot(epochs, t_loss, label=label, color=color, linewidth=2)
        axes[1].plot(epochs, v_miou, label=label, color=color, linewidth=2)

    axes[0].set_title("Training Loss vs. Epoch (Locked Budget)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].set_title("Validation mIoU (%) vs. Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation mIoU (%)"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Eval Precision: FP32 | Resolution: 256x256\n")

    val_ds = VOC2012SegmentationDataset("data/voc2012", split="val", img_size=256)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    models = {
        "Baseline U-Net (Slide 23)": (BaselineUNet(3, 21).to(device), "checkpoints/baseline_best.pt"),
        "Boundary-Gated U-Net (+Gated-SCNN)": (BoundaryGatedUNet(3, 21).to(device), "checkpoints/boundary_best.pt"),
    }

    results = {}
    for name, (model, ckpt) in models.items():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lat_b1, _ = measure_latency(model, device, batch_size=1)
        _, lat_b8_per_img = measure_latency(model, device, batch_size=8)
        per_cls, miou, acc, biou, records = evaluate_model(model, val_loader, device)
        results[name] = (params, lat_b1, lat_b8_per_img, per_cls, miou, acc, biou, records)

        print(f"=== {name} ===")
        print(f"Trainable Params: {params:,}")
        print(f"Latency (Batch=1): {lat_b1:.2f} ms/img | Latency (Batch=8): {lat_b8_per_img:.2f} ms/img")
        print(f"Val mIoU: {miou:.2f}% | Pixel Acc: {acc:.2f}% | 3px Boundary IoU: {biou:.2f}%\n")

    print("--- Per-Class IoU Comparison (%) ---")
    base_cls = results["Baseline U-Net (Slide 23)"][3]
    bnd_cls = results["Boundary-Gated U-Net (+Gated-SCNN)"][3]
    for idx, cls_name in enumerate(VOC_CLASSES):
        print(f"{idx:02d} {cls_name:<14} | Baseline: {base_cls[idx]:6.2f}% | Boundary-Gated: {bnd_cls[idx]:6.2f}% | Diff: {bnd_cls[idx]-base_cls[idx]:+6.2f}%")

    os.makedirs("outputs/failures", exist_ok=True)
    os.makedirs("outputs/successes", exist_ok=True)
    bnd_records = {r[1]: r for r in results["Boundary-Gated U-Net (+Gated-SCNN)"][7]}
    base_records = results["Baseline U-Net (Slide 23)"][7]

    for tag, subset in [("failures", base_records[:5]), ("successes", base_records[-5:])]:
        for iou_val, sid, img_t, gt_t, base_pred, _ in subset:
            _, _, _, _, bnd_pred, bnd_prob = bnd_records[sid]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255
            gt_vis = np.where(gt_t.numpy() == 255, 128, gt_t.numpy() * 12).astype(np.uint8)
            b_vis = (base_pred.numpy() * 12).astype(np.uint8)
            g_vis = (bnd_pred.numpy() * 12).astype(np.uint8)
            edge_vis = (bnd_prob.numpy() * 255).astype(np.uint8)

            panel = np.concatenate([
                rgb.astype(np.uint8),
                np.stack([gt_vis]*3, axis=-1),
                np.stack([b_vis]*3, axis=-1),
                np.stack([g_vis]*3, axis=-1),
                np.stack([edge_vis]*3, axis=-1),
            ], axis=1)
            Image.fromarray(panel).save(f"outputs/{tag}/{sid}_iou{iou_val:.2f}.png")

    plot_training_curves()
    print("\nSaved training_curves.png, failure panels, and success panels to outputs/.")

if __name__ == "__main__":
    main()
