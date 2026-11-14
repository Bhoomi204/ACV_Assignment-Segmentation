import csv, time, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import VOC2012SegmentationDataset
from src.models import BaselineUNet, BoundaryGatedUNet
from src.losses_metrics import SegmentationLoss, ConfusionMatrixMeter

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["baseline", "boundary"], required=True)
    parser.add_argument("--data_dir", default="data/voc2012")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = VOC2012SegmentationDataset(args.data_dir, split="train", img_size=256)
    val_ds = VOC2012SegmentationDataset(args.data_dir, split="val", img_size=256)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    model = (BaselineUNet(3, 21) if args.arch == "baseline" else BoundaryGatedUNet(3, 21)).to(device)
    criterion = SegmentationLoss(num_classes=21, ignore_index=255, bnd_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    log_path = f"logs/{args.arch}_train_log.csv"
    ckpt_path = f"checkpoints/{args.arch}_best.pt"
    best_miou = -1.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_miou", "val_pixel_acc", "epoch_sec"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            train_loss, steps = 0.0, 0

            for imgs, masks, bnd_gt, bnd_valid, _ in train_loader:
                if (masks != 255).sum() == 0:
                    continue
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                bnd_gt = bnd_gt.to(device, non_blocking=True)
                bnd_valid = bnd_valid.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    seg_logits, bnd_logits = model(imgs)
                    loss = criterion(seg_logits, bnd_logits, masks, bnd_gt, bnd_valid)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                steps += 1

            scheduler.step()

            model.eval()
            val_loss, val_steps = 0.0, 0
            meter = ConfusionMatrixMeter(num_classes=21, ignore_index=255)
            with torch.no_grad():
                for imgs, masks, bnd_gt, bnd_valid, _ in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    bnd_gt = bnd_gt.to(device, non_blocking=True)
                    bnd_valid = bnd_valid.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        seg_logits, bnd_logits = model(imgs)
                        loss = criterion(seg_logits, bnd_logits, masks, bnd_gt, bnd_valid)
                    val_loss += loss.item()
                    val_steps += 1
                    meter.update(torch.argmax(seg_logits, dim=1), masks)

            _, val_miou, val_acc = meter.compute()
            dt = time.time() - t0
            writer.writerow([epoch, f"{optimizer.param_groups[0]['lr']:.6f}", f"{train_loss/steps:.4f}",
                             f"{val_loss/val_steps:.4f}", f"{val_miou:.2f}", f"{val_acc:.2f}", f"{dt:.1f}"])
            f.flush()

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 5 == 0 or epoch == 1:
                print(f"[{args.arch.upper()}] Epoch {epoch:02d}/{args.epochs} ({dt:.1f}s) | "
                      f"TrainLoss={train_loss/steps:.4f} | ValmIoU={val_miou:.2f}% | Best={best_miou:.2f}%")

if __name__ == "__main__":
    main()
