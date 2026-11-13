"""Satu run pelatihan + evaluasi, dikendalikan oleh file konfigurasi JSON.

  python train.py --config runs/<run_id>/config.json

Kunci konfigurasi penting (lihat jalankan_eksperimen.py untuk nilai default):
  backbone, head, strategy (full|llrd|gradual_unfreeze|freeze_half), loss, lr, head_lr_mult,
  batch_size, grad_accum, dropout, max_epochs, patience, warmup_ratio, weight_decay, max_len,
  seed, normalisasi, train_files, val_files, test_sets, init_from, save_model, out_dir
"""
import argparse, json, os, time, random, math
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from transformers import get_linear_schedule_with_warmup
from modeling import (EmotionClassifier, load_tokenizer, BACKBONES, LABELS, L2I,
                      build_param_groups, apply_freeze, make_loss)
from normalisasi import Normalizer


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read(files, data_dir):
    d = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)
    d = d[d.label.isin(LABELS)].dropna(subset=['text'])
    return d.reset_index(drop=True)


class Batcher:
    def __init__(self, tok, max_len, norm):
        self.tok, self.max_len, self.norm = tok, max_len, norm
    def encode(self, texts):
        if self.norm:
            texts = [self.norm(t) for t in texts]
        return self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors='pt')
    def iterate(self, df, bs, shuffle, rng=None):
        idx = np.arange(len(df))
        if shuffle:
            rng.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = df.iloc[idx[k:k + bs]]
            enc = self.encode(b.text.astype(str).tolist())
            y = torch.tensor([L2I[l] for l in b.label])
            yield enc, y, b


def to_dev(enc, dev):
    return {k: v.to(dev) for k, v in enc.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}


@torch.no_grad()
def predict(model, batcher, df, dev, bs, amp):
    model.eval(); out = []
    for enc, y, _ in batcher.iterate(df, bs, False):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out.append(model(**to_dev(enc, dev)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, len(LABELS))


def ece(probs, y, bins=15):
    conf, pred = probs.max(1); acc = (pred == y).float(); e = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e


def fit_temperature(logits, y):
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); l = nll(logits / t.clamp(min=0.05), y); l.backward(); return l
    opt.step(closure)
    return float(t.clamp(min=0.05).item())


def metrics(logits, y, T=1.0):
    probs = torch.softmax(logits / T, -1); pred = probs.argmax(1)
    yt, yp = y.numpy(), pred.numpy()
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(len(LABELS)), zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
    present = sorted(set(yt.tolist()))
    emo = [i for i in present if LABELS[i] != 'neutral']
    return {
        'accuracy': accuracy_score(yt, yp),
        'precision_weighted': pw, 'recall_weighted': rw, 'f1_weighted': fw,
        'f1_macro': f1_score(yt, yp, labels=present, average='macro', zero_division=0),
        'f1_macro_emosi': f1_score(yt, yp, labels=emo, average='macro', zero_division=0) if emo else None,
        'nll': float(nn.functional.cross_entropy(logits / T, y).item()),
        'ece': ece(probs, y),
        'per_kelas': {LABELS[i]: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(s[i])}
                      for i in range(len(LABELS))},
        'confusion_matrix': confusion_matrix(yt, yp, labels=range(len(LABELS))).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); a = ap.parse_args()
    C = json.load(open(a.config))
    out = C['out_dir']; os.makedirs(out, exist_ok=True)
    set_seed(C['seed'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda' and C.get('fp16', True)
    t_start = time.time()

    bpath = C.get('backbone_path') or BACKBONES[C['backbone']]
    tok = load_tokenizer(bpath)
    norm = Normalizer(C['lexicon']) if C.get('normalisasi') else None
    batcher = Batcher(tok, C['max_len'], norm)

    train = read(C['train_files'], C['data_dir'])
    val = read(C['val_files'], C['data_dir'])
    tests = {name: read(files, C['data_dir']) for name, files in C['test_sets'].items()}

    model = EmotionClassifier(bpath, C['head'], C['dropout'], grad_ckpt=C.get('grad_ckpt', False))
    if C.get('init_from'):
        sd = torch.load(os.path.join(C['init_from'], 'model.pt'), map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
    model.to(dev)

    counts = train.label.value_counts().reindex(LABELS).fillna(0).values
    w = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
    loss_fn = make_loss(C['loss'], w.to(dev))

    strategy = C['strategy']
    head_lr = C['lr'] * C.get('head_lr_mult', 1.0)
    extra_epochs = 3 if strategy == 'gradual_unfreeze' else 0
    n_epochs = C['max_epochs'] + extra_epochs
    steps_per_epoch = math.ceil(len(train) / (C['batch_size'] * C['grad_accum']))
    opt_cls = torch.optim.AdamW
    if C.get('optim_8bit') and dev.type == 'cuda':
        try:
            import bitsandbytes as bnb
            opt_cls = bnb.optim.AdamW8bit
        except ImportError:
            print('PERINGATAN: bitsandbytes tidak terpasang, memakai AdamW biasa (risiko OOM untuk model large)')
    opt = opt_cls(build_param_groups(model, strategy, C['lr'], head_lr, C['weight_decay']), lr=C['lr'])
    sched = get_linear_schedule_with_warmup(opt, int(C['warmup_ratio'] * steps_per_epoch * n_epochs), steps_per_epoch * n_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    micro_bs = C['batch_size']
    rng = np.random.default_rng(C['seed'])

    best, best_state, bad, hist = float('inf'), None, 0, []
    for ep in range(n_epochs):
        n_blocks = apply_freeze(model, strategy, ep, n_epochs)
        model.train(); tl, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (enc, y, _) in enumerate(batcher.iterate(train, micro_bs, True, rng)):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(**to_dev(enc, dev))
            loss = loss_fn(logits.float(), y.to(dev)) / C['grad_accum']
            scaler.scale(loss).backward()
            if (step + 1) % C['grad_accum'] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            tl += loss.item() * C['grad_accum']; nb += 1
        if (step + 1) % C['grad_accum'] != 0:      # sisa akumulasi di akhir epoch
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        vlog = predict(model, batcher, val, dev, micro_bs * 2, amp)
        vy = torch.tensor([L2I[l] for l in val.label])
        vloss = nn.functional.cross_entropy(vlog, vy).item()
        vm = metrics(vlog, vy)
        hist.append({'epoch': ep + 1, 'train_loss': tl / max(nb, 1), 'val_loss': vloss,
                     'val_f1_weighted': vm['f1_weighted'], 'val_f1_macro': vm['f1_macro']})
        print(json.dumps(hist[-1]), flush=True)
        es_active = strategy != 'gradual_unfreeze' or ep >= n_blocks
        if vloss < best - 1e-4 or best_state is None:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
        elif es_active:
            bad += 1
            if bad >= C['patience']:
                break
    model.load_state_dict(best_state)

    # evaluasi + kalibrasi temperatur (dihitung di data validasi)
    vlog = predict(model, batcher, val, dev, micro_bs * 2, amp)
    vy = torch.tensor([L2I[l] for l in val.label])
    T = fit_temperature(vlog, vy)
    res = {'config': C, 'best_epoch': best_epoch, 'history': hist, 'temperature': T,
           'val': metrics(vlog, vy), 'val_kalibrasi': metrics(vlog, vy, T), 'test': {}, 'test_kalibrasi': {}}
    os.makedirs(os.path.join(out, 'prediksi'), exist_ok=True)
    for name, d in tests.items():
        lg = predict(model, batcher, d, dev, micro_bs * 2, amp)
        y = torch.tensor([L2I[l] for l in d.label])
        res['test'][name] = metrics(lg, y)
        res['test_kalibrasi'][name] = metrics(lg, y, T)
        pr = torch.softmax(lg / T, -1).numpy()
        pdf = d[['id', 'text', 'label']].copy(); pdf['pred'] = [LABELS[i] for i in pr.argmax(1)]
        for i, l in enumerate(LABELS):
            pdf[f'p_{l}'] = pr[:, i].round(4)
        pdf.to_csv(os.path.join(out, 'prediksi', f'{name}.csv'), index=False)
    res['durasi_detik'] = round(time.time() - t_start, 1)
    res['gpu'] = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    res['vram_puncak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == 'cuda' else 0
    if C.get('save_model'):
        induk = str(C.get('varian', '')).startswith('tahap1')      # dipakai sebagai init_from tahap 2
        sd = best_state if induk else {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
        torch.save(sd, os.path.join(out, 'model.pt'))
        tok.save_pretrained(os.path.join(out, 'tokenizer'))
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    print('SELESAI', out, json.dumps({k: round(v['f1_weighted'], 4) for k, v in res['test'].items()}))


if __name__ == '__main__':
    main()
