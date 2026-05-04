"""DBSCAN cluster-based pseudo-label extraction (better sample efficiency than top-K mutual NN).

Pipeline:
  1. Extract features from the proven 8-ckpt ensemble (7-baseline + cam-adv s500 ep60).
  2. Apply DBA(k=8) to gallery; compute rerank distance (k1=15, k2=4, lambda=0.275).
  3. Apply class-group filter (cross-class -> inf in the rerank dist).
  4. Build (Nq+Ng) square distance matrix:
       - intra-query and intra-gallery blocks set to 2.0 (max cosine distance)
         => we don't want DBSCAN to form intra-camera clusters; those don't help
            cross-camera pseudo-labeling.
       - cross-camera (q-g) block uses the reranked distance.
  5. Run sklearn DBSCAN with metric='precomputed' across a small eps sweep.
  6. For each cluster: keep only those that span >=1 query AND >=1 gallery
     (cross-camera identity groups). Drop noise (label=-1).
  7. Write pseudo_pairs.csv (cameraID, imageName, objectID, source, rerank_dist)
     in the same format used by build_pseudo_train_subset.py so we can reuse it.

We cache features+rerank_dist to .npz on first run; subsequent eps sweeps
reuse the cache (~3-5 sec per eps instead of re-extracting features).
"""
import csv
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


CAMADV_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
ALL_CKPTS = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
    CAMADV_EP60,
]
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
PSEUDO_ID_OFFSET = 1200

CACHE_PATH = '/workspace/miuam_challenge_diff/results/dbscan_features_cache.npz'
PAIRS_CSV  = '/workspace/miuam_challenge_diff/results/pseudo_pairs.csv'


def extract(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def build_or_load_cache(force_extract=False):
    """Returns (qf, gf, rrd, q_groups, g_groups, q_camids, q_names, g_camids, g_names)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    # ulimit on this host doesn't allow many forked workers; force serial dataloader
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    # Class-group setup
    CG = {'trafficsignal': 'trafficsignal', 'crosswalk': 'crosswalk',
          'container': 'bin_like', 'rubbishbins': 'bin_like'}
    qcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    q_groups = np.array([q2g[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g2g[os.path.basename(it[0])] for it in g_items])
    q_names  = np.array([os.path.basename(it[0]) for it in q_items])
    g_names  = np.array([os.path.basename(it[0]) for it in g_items])
    q_camids = np.array([it[2] for it in q_items])
    g_camids = np.array([it[2] for it in g_items])

    if (not force_extract) and os.path.exists(CACHE_PATH):
        print(f"  Loading cached features+rerank from {CACHE_PATH}")
        z = np.load(CACHE_PATH)
        return (z['qf'], z['gf'], z['rrd'], q_groups, g_groups, q_camids, q_names, g_camids, g_names)

    print("\n  Extracting 8-ckpt ensemble features:")
    qf_sum = gf_sum = None
    for ckpt in ALL_CKPTS:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        del model; torch.cuda.empty_cache()

    qf = F.normalize(qf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf_dba = db_augment(gf, DBA_K)

    print("\n  Computing rerank distance matrix...")
    q_g = np.dot(qf, gf_dba.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf_dba, gf_dba.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    # Class-group filter on rerank distance
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf

    print(f"  Saving cache to {CACHE_PATH}")
    np.savez(CACHE_PATH, qf=qf, gf=gf_dba, rrd=rrd)
    return (qf, gf_dba, rrd, q_groups, g_groups, q_camids, q_names, g_camids, g_names)


def run_dbscan(rrd, q_groups, g_groups, eps, min_samples=2):
    """Build square distance matrix [Nq+Ng, Nq+Ng] and run DBSCAN.
    Intra-query/gallery blocks set to a large finite distance (inf breaks DBSCAN).
    """
    Nq, Ng = rrd.shape
    N = Nq + Ng
    BIG = 10.0  # larger than any plausible eps; effectively disables intra-camera links
    D = np.full((N, N), BIG, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    # cross-camera block uses the rerank distance (post class-group filter)
    # Replace inf (cross-class) with BIG so DBSCAN doesn't blow up; same effect.
    rrd_safe = np.where(np.isfinite(rrd), rrd, BIG).astype(np.float32)
    D[:Nq, Nq:] = rrd_safe
    D[Nq:, :Nq] = rrd_safe.T

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=1)
    labels = db.fit_predict(D)
    return labels  # length N; -1 = noise


def summarize(labels, Nq, Ng):
    """Print cluster summary for a label array of length Nq+Ng."""
    q_lab, g_lab = labels[:Nq], labels[Nq:]
    cluster_ids = sorted(set(labels.tolist()) - {-1})
    n_total = len(cluster_ids)
    cross_camera = []
    for c in cluster_ids:
        nq = (q_lab == c).sum()
        ng = (g_lab == c).sum()
        if nq >= 1 and ng >= 1:
            cross_camera.append((c, int(nq), int(ng)))
    n_noise_q = int((q_lab == -1).sum())
    n_noise_g = int((g_lab == -1).sum())
    print(f"      total clusters: {n_total}, cross-camera: {len(cross_camera)}, "
          f"q-noise: {n_noise_q}/{Nq}, g-noise: {n_noise_g}/{Ng}")
    if cross_camera:
        sizes_q = [x[1] for x in cross_camera]
        sizes_g = [x[2] for x in cross_camera]
        print(f"      cross-camera cluster sizes: q={min(sizes_q)}-{max(sizes_q)} median {int(np.median(sizes_q))}, "
              f"g={min(sizes_g)}-{max(sizes_g)} median {int(np.median(sizes_g))}")
        print(f"      total pseudo-images: q={sum(sizes_q)}, g={sum(sizes_g)} -> {sum(sizes_q)+sum(sizes_g)} total")
    return cross_camera


def write_pseudo_csv(labels, cross_camera_clusters,
                     q_groups, g_groups, q_camids, q_names, g_camids, g_names,
                     rrd, out_path):
    """Write pseudo_pairs.csv compatible with build_pseudo_train_subset.py."""
    Nq = len(q_groups)
    Ng = len(g_groups)
    rows = []
    for pi, (cluster_id, _, _) in enumerate(cross_camera_clusters):
        pseudo_id = PSEUDO_ID_OFFSET + pi
        members = np.where(labels == cluster_id)[0]
        q_members = members[members < Nq]
        g_members = members[members >= Nq] - Nq
        # For each query in cluster, log row
        for qi in q_members:
            # rerank distance to the closest gallery member (for diagnostic)
            d_self = float(rrd[qi, g_members].min()) if len(g_members) else float('inf')
            rows.append({
                'cameraID': f'c{int(q_camids[qi]):03d}',
                'imageName': q_names[qi],
                'objectID': pseudo_id,
                'source': 'query',
                'rerank_dist': d_self,
            })
        for gi in g_members:
            d_self = float(rrd[q_members, gi].min()) if len(q_members) else float('inf')
            rows.append({
                'cameraID': f'c{int(g_camids[gi]):03d}',
                'imageName': g_names[gi],
                'objectID': pseudo_id,
                'source': 'gallery',
                'rerank_dist': d_self,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_clusters = len(cross_camera_clusters)
    print(f"\n  -> Wrote {out_path}: {len(rows)} rows, {n_clusters} pseudo-identities")
    return rows


def main():
    force = '--force-extract' in sys.argv
    qf, gf, rrd, q_groups, g_groups, q_camids, q_names, g_camids, g_names = build_or_load_cache(force_extract=force)
    Nq, Ng = rrd.shape
    print(f"\n  Nq={Nq}, Ng={Ng}, classes per side: q={dict(pd.Series(q_groups).value_counts())}, g={dict(pd.Series(g_groups).value_counts())}")
    finite = rrd[np.isfinite(rrd)]
    print(f"  rerank distance stats (finite, post class-group filter): "
          f"min={finite.min():.4f}, p1={np.percentile(finite,1):.4f}, "
          f"p5={np.percentile(finite,5):.4f}, p10={np.percentile(finite,10):.4f}, "
          f"median={np.median(finite):.4f}, max={finite.max():.4f}")

    # Sweep eps to see which gives a useful cluster count
    target_eps = None  # set by env var DBSCAN_EPS or default to mid-sweep value
    env_eps = os.environ.get('DBSCAN_EPS')
    sweep_eps = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    if env_eps is not None:
        sweep_eps = [float(env_eps)]

    best = None
    print("\n  DBSCAN sweep (min_samples=2):")
    for eps in sweep_eps:
        print(f"    eps={eps}:")
        labels = run_dbscan(rrd, q_groups, g_groups, eps=eps, min_samples=2)
        cc = summarize(labels, Nq, Ng)
        n_pseudo_imgs = sum(x[1] + x[2] for x in cc)
        # Heuristic: pick the eps that maximizes (cross-camera clusters * sqrt(images per cluster))
        if cc:
            score = len(cc) * (n_pseudo_imgs / max(1, len(cc))) ** 0.5
            if best is None or score > best[0]:
                best = (score, eps, labels, cc)

    if best is None:
        print("\n  ! No eps produced cross-camera clusters")
        return
    score, eps_star, labels_star, cc_star = best
    print(f"\n  Selected eps={eps_star} (heuristic score={score:.1f}, "
          f"{len(cc_star)} clusters, {sum(x[1]+x[2] for x in cc_star)} pseudo-images)")

    write_pseudo_csv(labels_star, cc_star, q_groups, g_groups,
                     q_camids, q_names, g_camids, g_names, rrd, PAIRS_CSV)


if __name__ == '__main__':
    main()
