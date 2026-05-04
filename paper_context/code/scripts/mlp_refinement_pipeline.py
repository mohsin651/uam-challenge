"""Post-hoc MLP refinement head on frozen 8-ckpt ensemble features.

Pipeline:
  1. Extract 8-ckpt sum features for Urban2026 train + query + gallery (cache to disk)
  2. Train a tiny MLP head (1024 → 512 → 1024) with triplet+CE loss on TRAIN
     features+labels, 50 epochs, ~10 min
  3. Inference: pass query+gallery features through trained MLP, then DBA + rerank
     + class-group filter

Generates ONE primary CSV (high-EV submission). Features are cached so re-runs
skip extraction.
"""
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from data.transforms.build import build_transforms
from data.common import CommDataset
from data.datasets import DATASET_REGISTRY
from model import make_model
from utils.re_ranking import re_ranking


CAMADV_S500_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
ALL_CKPTS = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
    CAMADV_S500_EP60,
]
CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache'
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275


# -------------------------- Feature extraction (cached) --------------------------
def _extract_one_loader(model, loader):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    return torch.cat(feats, 0)


def build_train_loader_for_inference():
    """Build a non-augmented train loader (test transforms applied to train data)."""
    test_transforms = build_transforms(cfg, is_train=False)
    dataset = DATASET_REGISTRY.get('UrbanElementsReID')(root=cfg.DATASETS.ROOT_DIR,
                                                       combineall=False)
    # Mimic the train-loader's domain-info wrapping
    train_items = [(it[0], it[1], it[2], {'domains': 0, 'q_or_g': 'train'}) for it in dataset.train]
    train_set = CommDataset(train_items, test_transforms, relabel=False)
    return DataLoader(train_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                      num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)


def extract_or_load():
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_path = f'{CACHE_DIR}/8ckpt_train.npz'
    test_path = f'{CACHE_DIR}/8ckpt_test.npz'

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        print(f"  loaded cache: train {train_data['feats'].shape}, test {test_data['qf'].shape}+{test_data['gf'].shape}")
        return (train_data['feats'], train_data['labels'],
                test_data['qf'], test_data['gf'], int(test_data['num_query']))

    # Build loaders
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    train_loader = build_train_loader_for_inference()
    train_items = train_loader.dataset.img_items
    train_labels = np.array([it[1] for it in train_items])

    # Extract per-ckpt and sum
    print(f"  num_query: {num_query}, gallery: {len(val_loader.dataset.img_items) - num_query}, train: {len(train_items)}")
    qf_sum = gf_sum = train_sum = None
    for ckpt in ALL_CKPTS:
        print(f"  extracting from {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        # query+gallery
        test_feats = _extract_one_loader(model, val_loader)
        qf, gf = test_feats[:num_query], test_feats[num_query:]
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        # train
        train_feats = _extract_one_loader(model, train_loader)
        train_sum = train_feats if train_sum is None else train_sum + train_feats
        del model; torch.cuda.empty_cache()

    # Re-normalize summed features
    qf = F.normalize(qf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_sum, p=2, dim=1).numpy().astype(np.float32)
    train_arr = F.normalize(train_sum, p=2, dim=1).numpy().astype(np.float32)

    np.savez(train_path, feats=train_arr, labels=train_labels)
    np.savez(test_path, qf=qf, gf=gf, num_query=num_query)
    print(f"  cached: {train_path}, {test_path}")
    return train_arr, train_labels, qf, gf, num_query


# -------------------------- MLP refinement head --------------------------
class RefineMLP(nn.Module):
    def __init__(self, dim=1024, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(h)
        # Residual + L2-normalize
        out = out + x
        return F.normalize(out, p=2, dim=1)


# -------------------------- Triplet+CE training --------------------------
def soft_triplet(feats, labels):
    """Soft-margin batch-hard triplet on L2-normalized features."""
    sim = feats @ feats.t()
    label_eq = labels[:, None] == labels[None, :]
    eye = torch.eye(feats.size(0), dtype=torch.bool, device=feats.device)
    pos_mask = label_eq & ~eye
    neg_mask = ~label_eq

    pos_sim = sim.masked_fill(~pos_mask, float('inf'))
    hardest_pos = pos_sim.min(dim=1)[0]
    neg_sim = sim.masked_fill(~neg_mask, float('-inf'))
    hardest_neg = neg_sim.max(dim=1)[0]
    return F.softplus(hardest_neg - hardest_pos).mean()


def pk_batch(feats, labels, P=4, K=16):
    """Sample P instances from K random identities. Returns (P*K, dim) feats and labels tensors."""
    unique_pids = np.unique(labels)
    pids = np.random.choice(unique_pids, K, replace=False)
    idxs = []
    for pid in pids:
        avail = np.where(labels == pid)[0]
        chosen = np.random.choice(avail, P, replace=len(avail) < P)
        idxs.extend(chosen)
    idxs = np.array(idxs)
    return feats[idxs], labels[idxs]


def train_mlp(train_feats, train_labels, num_classes, epochs=50,
              iters_per_epoch=200, lr=1e-3):
    device = torch.device('cuda')
    mlp = RefineMLP(dim=1024, hidden=512).to(device)
    classifier = nn.Linear(1024, num_classes, bias=False).to(device)
    nn.init.kaiming_normal_(classifier.weight)

    optim = torch.optim.Adam(list(mlp.parameters()) + list(classifier.parameters()),
                             lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * iters_per_epoch)

    train_feats_t = torch.from_numpy(train_feats)
    print(f"  training MLP: {epochs} epochs × {iters_per_epoch} iters, LR={lr}")

    for ep in range(epochs):
        mlp.train(); classifier.train()
        ep_trip = ep_ce = ep_acc = 0.0
        for it in range(iters_per_epoch):
            x_np, y_np = pk_batch(train_feats, train_labels, P=4, K=16)
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).long().to(device)
            x_refined = mlp(x)               # (PK, 1024) L2-normed
            logits = classifier(x_refined) * 16.0  # cosine softmax with temperature
            ce = F.cross_entropy(logits, y)
            trip = soft_triplet(x_refined, y)
            loss = trip + ce
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            ep_trip += trip.item(); ep_ce += ce.item()
            ep_acc += (logits.argmax(1) == y).float().mean().item()
        n = iters_per_epoch
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ep {ep+1:2d}/{epochs}  trip {ep_trip/n:.4f}  ce {ep_ce/n:.4f}  acc {ep_acc/n:.3f}  lr {sched.get_last_lr()[0]:.2e}")

    mlp.eval()
    return mlp


# -------------------------- DBA + rerank + write --------------------------
def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(qf, gf, label, num_query, val_loader):
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    q_groups = np.array([q2g[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g2g[os.path.basename(it[0])] for it in g_items])
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf

    indices = np.argsort(rrd, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    print("\n[1/3] Extract / load 8-ckpt features (train + query + gallery)")
    train_feats, train_labels, qf_arr, gf_arr, num_query = extract_or_load()
    num_classes = int(train_labels.max() + 1)
    print(f"  num_classes: {num_classes}, train: {train_feats.shape}, query: {qf_arr.shape}, gallery: {gf_arr.shape}")

    print("\n[2/3] Train MLP refinement head")
    torch.manual_seed(2026)
    np.random.seed(2026)
    mlp = train_mlp(train_feats, train_labels, num_classes=num_classes,
                    epochs=50, iters_per_epoch=200, lr=1e-3)

    print("\n[3/3] Apply MLP at inference + write CSV")
    val_loader, _ = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    with torch.no_grad():
        q_in = torch.from_numpy(qf_arr).cuda()
        g_in = torch.from_numpy(gf_arr).cuda()
        q_refined = mlp(q_in).cpu().numpy().astype(np.float32)
        g_refined = mlp(g_in).cpu().numpy().astype(np.float32)
    write_csv(q_refined, g_refined, 'mlp_refine_baseline8', num_query, val_loader)


if __name__ == '__main__':
    main()
