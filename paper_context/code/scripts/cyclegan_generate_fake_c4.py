"""Generate fake-c004 versions of all 11,175 c001-c003 training images.

Loads the trained CycleGAN generator G_A (c001-c003 -> c004) and applies it
to every training image. Output: /workspace/Urban2026_cyclegan/image_train_fake_c4/
with one synthetic image per real training image (same filename + 'fake_c4_' prefix
inside the merged dataset folder later).

Run AFTER CycleGAN training is complete.
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, '/workspace/cyclegan')
from models import networks


CHECKPOINT_DIR = Path('/workspace/cyclegan_checkpoints/urban_c003_to_c004')
SRC_DIR = Path('/workspace/Urban2026/image_train')
DST_DIR = Path('/workspace/Urban2026_cyclegan/image_train_fake_c4')
TARGET_SIZE = (128, 256)  # (W, H), matches PAT input


def load_generator(ckpt_name='latest_net_G_A.pth'):
    netG = networks.define_G(
        input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
        norm='instance', use_dropout=False, init_type='normal', init_gain=0.02,
    )
    state = torch.load(CHECKPOINT_DIR / ckpt_name, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    netG.load_state_dict(state)
    netG = netG.cuda().eval()
    return netG


@torch.no_grad()
def translate(netG, img_path: Path) -> Image.Image:
    """Apply G_A: real c001-c003 -> fake c004."""
    img = Image.open(img_path).convert('RGB').resize(TARGET_SIZE, Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    arr = arr * 2.0 - 1.0  # [-1, 1] (CycleGAN range)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()  # [1, 3, H, W]
    out = netG(t).cpu().squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]
    out = (out + 1.0) / 2.0
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='latest_net_G_A.pth',
                        help='which generator checkpoint to use')
    parser.add_argument('--limit', type=int, default=0,
                        help='if >0, only translate first N images (for sample inspection)')
    parser.add_argument('--dst', default=str(DST_DIR),
                        help='output directory')
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(SRC_DIR))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"Loading G_A from {CHECKPOINT_DIR}/{args.ckpt}")
    netG = load_generator(args.ckpt)
    print(f"Translating {len(files)} images to {dst}/")
    for i, f in enumerate(files):
        out = translate(netG, SRC_DIR / f)
        out.save(dst / f, 'JPEG', quality=95)
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(files)}")
    print(f"Done. Wrote {len(files)} synthetic c004 images.")


if __name__ == '__main__':
    main()
