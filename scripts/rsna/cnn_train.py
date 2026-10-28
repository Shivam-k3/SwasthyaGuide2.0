"""Train the DenseNet121 pneumonia baseline on the processed RSNA cache.

Usage:
  python scripts/rsna/cnn_train.py                       # full defaults
  python scripts/rsna/cnn_train.py --epochs 20 --batch-size 32 --lr 1e-4
  python scripts/rsna/cnn_train.py --no-pretrained --limit-train 8 ...  # smoke

Pipeline: load verified cache -> exclude unlabelled -> assert patient-level
split separation -> train with BCEWithLogitsLoss (optional pos_weight) ->
select threshold on VALIDATION (Youden J) -> evaluate TEST once, with the
best checkpoint and the chosen threshold. The test set is never used for
model or threshold selection.

Outputs (under --out-dir, default <RSNA_BASE>/runs/cnn_densenet121_<ts>/):
  config.json, history.json, best.pt, last.pt, threshold.json,
  val_metrics.json, test_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn_data
import cnn_metrics
import cnn_model  # noqa: F401  (imported by callers; keeps path setup symmetric)

DEFAULT_BASE = os.environ.get('RSNA_BASE', r'D:\Datasets\RSNA_Pneumonia')


@dataclass
class TrainConfig:
    rsna_base: str
    out_dir: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    device: str = 'auto'          # auto | cpu | cuda
    class_weight: str = 'none'    # none | pos_weight
    pretrained: bool = True
    amp: bool = False             # mixed precision (CUDA only; ignored on CPU)
    limit_train: int | None = None    # per-class caps, for smoke runs only
    limit_val: int | None = None
    limit_test: int | None = None


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(name)
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    return dev


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    g = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, generator=g,
                      pin_memory=(cfg.device == 'cuda'))


def run_epoch(model, loader, optimizer, loss_fn, device, amp: bool = False) -> float:
    """One training pass; returns the mean loss. Raises on non-finite loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=amp and device.type == 'cuda'):
            logits = model(x)
            loss = loss_fn(logits, y)
        if not math.isfinite(loss.detach()):
            raise RuntimeError(f'non-finite training loss ({float(loss)}); '
                               f'check inputs, lr and class weights')
        loss.backward()
        optimizer.step()
        total += float(loss) * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_prob) with probabilities via sigmoid."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _bce_from_probs(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _atomic_save(obj, path: str) -> None:
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model(model, loaders: dict, cfg: TrainConfig, out_dir: str,
                pos_weight: float | None = None,
                extra_config: dict | None = None) -> list[dict]:
    """Train/evaluate loop. Model is injected so tests can pass a tiny net.

    Writes config.json, history.json, best.pt, last.pt, threshold.json,
    val_metrics.json and test_metrics.json under out_dir.
    """
    device = resolve_device(cfg.device)
    model = model.to(device)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**asdict(cfg), **(extra_config or {})}, f, indent=2)
    loss_fn = (nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
        if pos_weight is not None else nn.BCEWithLogitsLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict] = []
    best_auroc, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(model, loaders['train'], optimizer, loss_fn, device, cfg.amp)
        y_val, p_val = evaluate(model, loaders['validation'], device)
        val_loss = _bce_from_probs(y_val, p_val)
        vm = cnn_metrics.compute_metrics(y_val, p_val, 0.5)   # @0.5 for history
        row = {'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss,
               'val_auroc': vm['auroc'], 'lr': cfg.lr,
               'seconds': round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {epoch:>3}/{cfg.epochs}  train_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f}  val_auroc={vm['auroc']:.4f}", flush=True)
        if vm['auroc'] > best_auroc:
            best_auroc, best_epoch = vm['auroc'], epoch
            _atomic_save({'state_dict': model.state_dict(), 'epoch': epoch,
                          'val_auroc': vm['auroc'],
                          'config': asdict(cfg)}, os.path.join(out_dir, 'best.pt'))
        _atomic_save({'state_dict': model.state_dict(), 'epoch': epoch,
                      'config': asdict(cfg)}, os.path.join(out_dir, 'last.pt'))
        with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    # threshold selection: VALIDATION ONLY, using the best checkpoint
    ckpt = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    y_val, p_val = evaluate(model, loaders['validation'], device)
    threshold = cnn_metrics.select_threshold_youden(y_val, p_val)
    val_metrics = cnn_metrics.compute_metrics(y_val, p_val, threshold)
    with open(os.path.join(out_dir, 'threshold.json'), 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'selected_on': 'validation',
                   'criterion': 'max Youden J (sensitivity + specificity - 1)',
                   'best_epoch': ckpt['epoch']}, f, indent=2)
    with open(os.path.join(out_dir, 'val_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(val_metrics, f, indent=2)

    # final evaluation: TEST touched exactly once, after all selection
    y_test, p_test = evaluate(model, loaders['test'], device)
    test_metrics = cnn_metrics.compute_metrics(y_test, p_test, threshold)
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"best epoch {ckpt['epoch']} (val_auroc={ckpt['val_auroc']:.4f}); "
          f"threshold {threshold:.4f} (validation, Youden J)")
    print(f"TEST: auroc={test_metrics['auroc']:.4f} "
          f"sens={test_metrics['sensitivity']:.4f} "
          f"spec={test_metrics['specificity']:.4f} "
          f"precision={test_metrics['precision']:.4f} "
          f"cm={test_metrics['confusion_matrix']['matrix']}")
    return history


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rsna-base', default=DEFAULT_BASE,
                   help='dataset root (default $RSNA_BASE)')
    p.add_argument('--out-dir', default=None,
                   help='output dir (default <base>/runs/cnn_densenet121_<timestamp>)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--class-weight', choices=['none', 'pos_weight'], default='none',
                   help="pos_weight = n_negative/n_positive in BCEWithLogitsLoss")
    p.add_argument('--no-pretrained', action='store_true',
                   help='random init (smoke tests; never for the real baseline)')
    p.add_argument('--amp', action='store_true', help='mixed precision (CUDA only)')
    p.add_argument('--limit-train', type=int, default=None, metavar='N',
                   help='smoke only: first N per class in train')
    p.add_argument('--limit-val', type=int, default=None, metavar='N')
    p.add_argument('--limit-test', type=int, default=None, metavar='N')
    args = p.parse_args(argv)

    out_dir = args.out_dir or os.path.join(
        args.rsna_base, 'runs',
        'cnn_densenet121_' + time.strftime('%Y%m%d_%H%M%S'))
    cfg = TrainConfig(rsna_base=args.rsna_base, out_dir=out_dir, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                      num_workers=args.num_workers, device=args.device,
                      class_weight=args.class_weight, pretrained=not args.no_pretrained,
                      amp=args.amp, limit_train=args.limit_train,
                      limit_val=args.limit_val, limit_test=args.limit_test)

    seed_everything(cfg.seed)
    datasets, rows = cnn_data.make_datasets(cfg.rsna_base)
    for split, cap in (('train', cfg.limit_train), ('validation', cfg.limit_val),
                       ('test', cfg.limit_test)):
        if cap is not None:
            rows[split] = cnn_data.stratified_limit(rows[split], cap)
            datasets[split] = cnn_data.RSNADataset(
                os.path.join(cfg.rsna_base, 'proc'), rows[split])
            print(f'{split}: limited to {len(rows[split])} images (smoke run)')
    n_pos, n_neg = cnn_data.label_counts(rows['train'])
    print(f'train: {n_pos:,} positive / {n_neg:,} negative '
          f'({n_pos / (n_pos + n_neg) * 100:.2f}% positive)')

    loaders = {s: make_loader(d, cfg, shuffle=(s == 'train'))
               for s, d in datasets.items()}
    model = cnn_model.build_model(pretrained=cfg.pretrained)
    pos_weight = (cnn_data.pos_weight_value(rows['train'])
                  if cfg.class_weight == 'pos_weight' else None)
    if pos_weight is not None:
        print(f'class weighting: pos_weight={pos_weight:.4f}')

    device = resolve_device(cfg.device)
    print(f'device: {device} | pretrained: {cfg.pretrained} | out: {out_dir}')
    extra = {'software': {'python': sys.version.split()[0],
                          'torch': torch.__version__,
                          'numpy': np.__version__},
             'train_counts': {'positive': n_pos, 'negative': n_neg}}

    train_model(model, loaders, cfg, out_dir, pos_weight=pos_weight,
                extra_config=extra)
    print('done; artifacts in', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
