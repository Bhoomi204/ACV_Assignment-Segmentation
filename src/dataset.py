import os, random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

def _build_voc_rgb_lut():
    lut = np.full(1 << 24, 255, dtype=np.int64)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        idx = (r << 16) | (g << 8) | b
        lut[idx] = i if (i < 21 or i == 255) else 255
    return lut

VOC_RGB_LUT = _build_voc_rgb_lut()

def load_voc_mask_as_2d(mask_pil: Image.Image) -> Image.Image:
    if mask_pil.mode in ("P", "L"):
        return mask_pil
    arr = np.array(mask_pil.convert("RGB"), dtype=np.int64)
    if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
        ids = arr[..., 0]
    else:
        rgb_key = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
        ids = VOC_RGB_LUT[rgb_key]
    return Image.fromarray(ids.astype(np.uint8), mode="L")

def extract_ignore_aware_boundary(mask_tensor: torch.Tensor):
    valid = (mask_tensor != 255)
    clean_4d = torch.where(valid, mask_tensor, torch.zeros_like(mask_tensor)).float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(clean_4d, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-clean_4d, kernel_size=3, stride=1, padding=1)
    raw_edge = (dilated != eroded).squeeze(0).squeeze(0).float()

    valid_4d = valid.float().unsqueeze(0).unsqueeze(0)
    valid_eroded = (-F.max_pool2d(-valid_4d, kernel_size=3, stride=1, padding=1)).squeeze(0).squeeze(0)
    return raw_edge * valid_eroded, valid_eroded

class VOC2012SegmentationDataset(Dataset):
    """Pre-caches resized uint8 tensors in RAM once so GPU training runs at full speed."""
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.ids = sorted([f.split(".")[0] for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        self.split = split
        self.img_size = img_size
        self.cache_size = 288 if split == "train" else img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.cached_imgs = []
        self.cached_masks = []
        for sid in self.ids:
            img = Image.open(os.path.join(self.img_dir, f"{sid}.jpg")).convert("RGB")
            mask = load_voc_mask_as_2d(Image.open(os.path.join(self.mask_dir, f"{sid}.png")))
            img = TF.resize(img, [self.cache_size, self.cache_size], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.cache_size, self.cache_size], interpolation=InterpolationMode.NEAREST)
            self.cached_imgs.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))
            self.cached_masks.append(torch.from_numpy(np.array(mask, dtype=np.uint8)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_u8 = self.cached_imgs[idx]
        mask_u8 = self.cached_masks[idx]

        if self.split == "train":
            if random.random() < 0.5:
                img_u8 = torch.flip(img_u8, dims=[2])
                mask_u8 = torch.flip(mask_u8, dims=[1])
            max_off = self.cache_size - self.img_size
            top = random.randint(0, max_off)
            left = random.randint(0, max_off)
            img_u8 = img_u8[:, top:top + self.img_size, left:left + self.img_size]
            mask_u8 = mask_u8[top:top + self.img_size, left:left + self.img_size]

        img_tensor = (img_u8.float().div_(255.0) - self.mean) / self.std
        mask_tensor = mask_u8.long()
        bnd_target, bnd_valid = extract_ignore_aware_boundary(mask_tensor)
        return img_tensor, mask_tensor, bnd_target, bnd_valid, self.ids[idx]
