"""Runner eksperimen bertahap (turnamen) untuk mencari konfigurasi IndoBERT terbaik.

Pemakaian (dari folder eksperimen/):
  python jalankan_eksperimen.py                 # jalankan semua fase berurutan
  python jalankan_eksperimen.py --fase 1        # hanya fase tertentu (1-6)
  python jalankan_eksperimen.py --semua-backbone  # fase 2 memakai 10 backbone, bukan 3 terbaik
  python jalankan_eksperimen.py --rencana       # hanya tampilkan daftar run tanpa melatih

Run yang sudah selesai (ada metrics.json) otomatis dilewati, sehingga aman dihentikan
dan dilanjutkan kapan saja. Semua hasil terkumpul di hasil/semua_run.csv.

Fase:
 1 backbone      : 10 varian IndoBERT x 3 learning rate, head [CLS], full fine-tuning (tahap 1)
 2 arsitektur    : backbone terbaik x 5 head x 4 strategi fine-tuning (tahap 1)
 3 loss          : 3 konfigurasi terbaik x 4 fungsi loss (tahap 1)
 4 hyperparameter: konfigurasi terbaik x grid proposal (lr x batch x dropout) (tahap 1)
 5 tahap2        : 3 seed x {satu tahap, tahap-2 saja, dua tahap x 8 varian}
 6 final         : ablasi normalisasi, ablasi sumber neutral, pembanding IndoBERT-base-p1
Seleksi fase 1-4 memakai weighted F1 validasi tahap 1 (sesuai proposal); seleksi fase 5-6
memakai macro F1 validasi domain konseling. Data uji tidak pernah dipakai untuk memilih.
"""
import argparse, json, os, subprocess, sys, itertools, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('TA_DATA_DIR', os.path.join(HERE, '..', 'data_final'))
RUNS = os.environ.get('TA_RUNS_DIR', os.path.join(HERE, 'runs'))
HASIL = os.environ.get('TA_HASIL_DIR', os.path.join(HERE, 'hasil'))
LEX = os.path.join(HERE, 'colloquial-indonesian-lexicon.csv')

sys.path.insert(0, HERE)
from modeling import BACKBONES  # noqa

LARGE = {'indobert-large-p1', 'indobert-large-p2'}
BASE_OVERRIDE = json.loads(os.environ.get('TA_BASE_OVERRIDE', '{}'))  # mis. {"max_epochs": 1} untuk uji cepat
BASE = dict(head='cls', strategy='full', loss='weighted_ce', lr=2e-5, head_lr_mult=1.0, batch_size=16,
            grad_accum=1, dropout=0.1, max_epochs=5, patience=2, warmup_ratio=0.1, weight_decay=0.01,
            max_len=128, seed=42, normalisasi=False, fp16=True, lexicon=LEX, data_dir=DATA)
T1 = dict(train_files=['tahap1_train.csv'], val_files=['tahap1_val.csv'],
          test_sets={'tahap1_test': ['tahap1_test.csv'], 'tahap2_test': ['tahap2_test.csv']})
T2 = dict(train_files=['tahap2_train.csv'], val_files=['tahap2_val.csv'],
          test_sets={'tahap2_test': ['tahap2_test.csv'], 'tahap1_test': ['tahap1_test.csv']})
SATU = dict(train_files=['satu_tahap_train.csv'], val_files=['satu_tahap_val.csv'],
            test_sets={'tahap2_test': ['tahap2_test.csv'], 'tahap1_test': ['tahap1_test.csv'],
                       'tahap2_val': ['tahap2_val.csv']})
SEEDS = [42, 43, 44]


def vram_gb():
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def hw_adjust(cfg):
    """Sesuaikan batch mikro / akumulasi / checkpointing agar muat di VRAM laptop.
    Batch efektif (batch_size x grad_accum) tetap sama dengan batch_size pada grid."""
    eff = cfg['batch_size'] * cfg['grad_accum']
    v = vram_gb()
    if cfg['backbone'] in LARGE:
        micro = 4 if v < 12 else 8
        cfg.update(batch_size=micro, grad_accum=max(1, eff // micro), grad_ckpt=True, optim_8bit=0 < v < 10)
    elif v and v < 5 and eff > 8:
        cfg.update(batch_size=8, grad_accum=eff // 8)
    cfg['batch_efektif'] = eff
    return cfg


def head_mult(head):
    return 20.0 if head.startswith('bilstm') else 1.0


def run_id(c, tag=''):
    parts = [c['fase'], c['backbone'], c['head'], c['strategy'], c['loss'], f"lr{c['lr']:.0e}",
             f"bs{c['batch_efektif']}", f"do{c['dropout']}", f"s{c['seed']}"]
    if tag:
        parts.append(tag)
    return '_'.join(str(p) for p in parts).replace('+', '').replace('-0', '-')


def make(fase, backbone, data, tag='', **kw):
    c = dict(BASE); c.update(BASE_OVERRIDE); c.update(data); c.update(kw); c['fase'] = fase; c['backbone'] = backbone
    c['head_lr_mult'] = kw.get('head_lr_mult', head_mult(c['head']))
    c = hw_adjust(c)
    c['run_id'] = run_id(c, tag)
    c['out_dir'] = os.path.join(RUNS, c['run_id'])
    return c


def done(c):
    return os.path.exists(os.path.join(c['out_dir'], 'metrics.json'))


def execute(cfgs, rencana=False):
    todo = [c for c in cfgs if not done(c)]
    print(f'  {len(cfgs)} run di fase ini, {len(cfgs) - len(todo)} sudah selesai, {len(todo)} akan dijalankan')
    if rencana:
        for c in todo:
            print('   -', c['run_id'])
        return
    for i, c in enumerate(todo, 1):
        os.makedirs(c['out_dir'], exist_ok=True)
        if c.get('init_from') and not os.path.exists(os.path.join(c['init_from'], 'model.pt')):
            print(f'  [lewati] checkpoint induk belum ada: {c["init_from"]}'); continue
        json.dump(c, open(os.path.join(c['out_dir'], 'config.json'), 'w'), indent=1)
        t = time.time()
        print(f'  [{i}/{len(todo)}] {c["run_id"]}', flush=True)
        with open(os.path.join(c['out_dir'], 'log.txt'), 'w') as log:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'train.py'), '--config',
                                os.path.join(c['out_dir'], 'config.json')], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        status = 'ok' if r.returncode == 0 else ('OOM' if 'out of memory' in open(os.path.join(c['out_dir'], 'log.txt')).read().lower() else 'gagal')
        if status != 'ok':
            json.dump({'status': status}, open(os.path.join(c['out_dir'], 'status.json'), 'w'))
        print(f'      -> {status}, {time.time() - t:.0f} detik', flush=True)
    kumpulkan()


def load(c):
    p = os.path.join(c['out_dir'], 'metrics.json')
    return json.load(open(p)) if os.path.exists(p) else None


def score(c, key='val', metric='f1_weighted'):
    m = load(c)
    return m[key][metric] if m else -1


def kumpulkan():
    rows = []
    for d in sorted(os.listdir(RUNS)) if os.path.exists(RUNS) else []:
        p = os.path.join(RUNS, d, 'metrics.json')
        if not os.path.exists(p):
            continue
        m = json.load(open(p)); c = m['config']
        r = {k: c.get(k) for k in ['run_id', 'fase', 'backbone', 'head', 'strategy', 'loss', 'lr', 'batch_efektif',
                                   'dropout', 'seed', 'normalisasi', 'varian', 'rehearsal', 'lr_tahap2']}
        r.update(best_epoch=m['best_epoch'], durasi_menit=round(m['durasi_detik'] / 60, 1), vram_gb=m.get('vram_puncak_gb'),
                 temperature=round(m['temperature'], 3),
                 val_f1_weighted=m['val']['f1_weighted'], val_f1_macro=m['val']['f1_macro'])
        for tn, tm in m['test'].items():
            for k in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro', 'f1_macro_emosi', 'ece']:
                r[f'{tn}_{k}'] = tm[k]
            r[f'{tn}_ece_kalibrasi'] = m['test_kalibrasi'][tn]['ece']
            for lab, v in tm['per_kelas'].items():
                r[f'{tn}_f1_{lab}'] = v['f1']
        rows.append(r)
    os.makedirs(HASIL, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(HASIL, 'semua_run.csv'), index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ definisi fase
def fase1():
    return [make('F1', b, T1, lr=lr) for b in BACKBONES for lr in [2e-5, 3e-5, 5e-5]]


def best_lr_per_backbone():
    best = {}
    for c in fase1():
        s = score(c)
        if s > best.get(c['backbone'], (None, -1))[1]:
            best[c['backbone']] = (c['lr'], s)
    return best


def fase2(semua=False, k=3):
    best = {b: v for b, v in best_lr_per_backbone().items() if v[1] >= 0}
    ranked = sorted(best, key=lambda b: -best[b][1])
    chosen = ranked if semua else ranked[:k]
    heads = ['cls', 'mean', 'cls_mean', 'bilstm', 'bilstm_attn']
    strategies = ['full', 'llrd', 'gradual_unfreeze', 'freeze_half']
    return [make('F2', b, T1, lr=best[b][0], head=h, strategy=s) for b in chosen for h in heads for s in strategies]


def top(cfgs, n, key='val', metric='f1_weighted'):
    scored = [(score(c, key, metric), c) for c in cfgs]
    return [c for s, c in sorted(scored, key=lambda x: -x[0]) if s >= 0][:n]


ARCH_KEYS = ['backbone', 'head', 'strategy', 'lr', 'head_lr_mult']


def fase3(semua=False):
    base = top(fase2(semua), 3)
    return [make('F3', c['backbone'], T1, loss=l, **{k: c[k] for k in ARCH_KEYS if k != 'backbone'})
            for c in base for l in ['ce', 'weighted_ce', 'focal', 'label_smoothing']]


def fase4(semua=False):
    b = top(fase3(semua), 1)
    if not b:
        return []
    b = b[0]
    keep = {k: b[k] for k in ['head', 'strategy', 'loss', 'head_lr_mult']}
    return [make('F4', b['backbone'], T1, lr=lr, batch_size=bs, dropout=do, **keep)
            for lr, bs, do in itertools.product([1e-5, 2e-5, 3e-5, 5e-5], [16, 32], [0.1, 0.2, 0.3])]


def best_stage1(semua=False):
    b = top(fase4(semua), 1)
    return b[0] if b else None


def stage1_seed(b, seed):
    keep = {k: b[k] for k in ['head', 'strategy', 'loss', 'lr', 'batch_size', 'dropout', 'head_lr_mult']}
    keep['batch_size'] = b['batch_efektif']
    return make('F5', b['backbone'], T1, tag='tahap1', seed=seed, save_model=True, varian='tahap1_saja', **keep)


def fase5(semua=False):
    b = best_stage1(semua)
    if b is None:
        return []
    arch = {k: b[k] for k in ['head', 'dropout', 'head_lr_mult']}
    cfgs = []
    for seed in SEEDS:
        s1 = stage1_seed(b, seed)
        cfgs.append(s1)
        # baseline satu tahap: gabungan data, konfigurasi sama
        cfgs.append(make('F5', b['backbone'], SATU, tag='satu_tahap', seed=seed, varian='satu_tahap', lr=b['lr'],
                         strategy=b['strategy'], loss=b['loss'], batch_size=b['batch_efektif'], **arch))
        # tahap 2 saja: IndoBERT pretrained langsung ke data konseling (tanpa tahap 1)
        cfgs.append(make('F5', b['backbone'], T2, tag='tahap2_saja', seed=seed, varian='tahap2_saja', lr=b['lr'],
                         strategy='full', loss='ce', batch_size=16, **arch))
        # dua tahap: 8 varian
        for lr2, strat2, reh in itertools.product([b['lr'], b['lr'] / 2], ['full', 'freeze_half'], [False, True]):
            data = dict(T2); data['train_files'] = T2['train_files'] + (['rehearsal_tahap1.csv'] if reh else [])
            tag = f"duatahap_lr2{'penuh' if lr2 == b['lr'] else 'setengah'}_{strat2}_{'reh' if reh else 'noreh'}"
            cfgs.append(make('F5', b['backbone'], data, tag=tag, seed=seed, init_from=s1['out_dir'], lr=lr2,
                             strategy=strat2, loss='ce', batch_size=16, varian='dua_tahap', rehearsal=reh, save_model=True,
                             lr_tahap2=lr2, **arch))
    return cfgs


def best_variant(semua=False):
    df = pd.DataFrame([{'tag': c['run_id'].split('_s')[-1].split('_', 1)[-1] if 'duatahap' in c['run_id'] else c['varian'],
                        'score': score(c, 'val', 'f1_macro'), 'cfg': c} for c in fase5(semua) if c['varian'] == 'dua_tahap'])
    if df.empty or (df.score < 0).all():
        return None
    g = df[df.score >= 0].groupby('tag').score.mean().sort_values(ascending=False)
    return g.index[0], [c for c in df.cfg if g.index[0] in c['run_id']]


def fase6(semua=False):
    b = best_stage1(semua); bv = best_variant(semua)
    if b is None or bv is None:
        return []
    tag, cfgs2 = bv
    cfgs = []
    for c2 in cfgs2:
        seed = c2['seed']
        base_kw = {k: c2[k] for k in ['head', 'dropout', 'head_lr_mult', 'lr', 'strategy', 'loss', 'rehearsal', 'lr_tahap2']}
        # (a) ablasi normalisasi: tahap 1 + tahap 2 dengan normalisasi teks
        s1 = stage1_seed(b, seed); s1n = dict(s1)
        s1n = make('F6', b['backbone'], T1, tag='tahap1_norm', seed=seed, save_model=True, normalisasi=True, varian='tahap1_norm',
                   **{k: s1[k] for k in ['head', 'strategy', 'loss', 'lr', 'dropout', 'head_lr_mult']}, batch_size=s1['batch_efektif'])
        cfgs.append(s1n)
        cfgs.append(make('F6', b['backbone'], {**T2, 'train_files': c2['train_files']}, tag='duatahap_norm', seed=seed,
                         init_from=s1n['out_dir'], normalisasi=True, batch_size=16, varian='dua_tahap_norm', **base_kw))
        # (b) ablasi sumber neutral: neutral tahap 2 dari tweet Riccosan (rancangan awal proposal)
        alt = {'train_files': ['tahap2alt_train.csv'] + (['rehearsal_tahap1.csv'] if c2.get('rehearsal') else []),
               'val_files': ['tahap2alt_val.csv'],
               'test_sets': {'tahap2_test': ['tahap2_test.csv'], 'tahap2alt_test': ['tahap2alt_test.csv'],
                             'tahap1_test': ['tahap1_test.csv']}}
        cfgs.append(make('F6', b['backbone'], alt, tag='duatahap_neutral_riccosan', seed=seed, init_from=s1['out_dir'],
                         batch_size=16, varian='neutral_riccosan', **base_kw))
        # (c) pembanding: IndoBERT-base-p1 (model di proposal) dengan pipeline terbaik
        if b['backbone'] != 'indobert-base-p1':
            p1_lr = best_lr_per_backbone().get('indobert-base-p1', (2e-5, -1))[0]   # lr terbaik base-p1 di fase 1
            p1 = make('F6', 'indobert-base-p1', T1, tag='tahap1_p1', seed=seed, save_model=True, varian='tahap1_p1',
                      **{k: b[k] for k in ['head', 'strategy', 'loss', 'dropout', 'head_lr_mult']}, lr=p1_lr, batch_size=b['batch_efektif'])
            cfgs.append(p1)
            cfgs.append(make('F6', 'indobert-base-p1', {**T2, 'train_files': c2['train_files']}, tag='duatahap_p1', seed=seed,
                             init_from=p1['out_dir'], batch_size=16, varian='dua_tahap_p1', **{**base_kw, 'lr': p1_lr * c2['lr'] / b['lr']}))
    return cfgs


def fase7(semua=False):
    """Pembanding adil untuk pertanyaan penelitian 1. Di fase 5 varian dua tahap dipilih dari beberapa
    kombinasi learning rate dan strategi, sedangkan tiap pembanding hanya punya satu konfigurasi. Di sini
    pembanding diberi ruang pencarian yang sama (learning rate penuh/setengah x full/freeze_half) dan
    dipilih dengan cara yang sama, yaitu macro F1 validasi domain konseling.
      (a) tahap 2 saja  : IndoBERT pretrained langsung ke data konseling
      (b) satu tahap    : data gabungan, early stopping pada validasi domain konseling
      (c) head [CLS] + full fine-tuning (arsitektur rancangan proposal) dengan pipeline dua tahap terpilih
    """
    b = best_stage1(semua); bv = best_variant(semua)
    if b is None or bv is None:
        return []
    tag2, cfgs2 = bv
    arch = {k: b[k] for k in ['head', 'dropout', 'head_lr_mult']}
    satu_val = dict(SATU); satu_val['val_files'] = ['tahap2_val.csv']
    cfgs = []
    for seed in SEEDS:
        for lr, strat in itertools.product([b['lr'], b['lr'] / 2], ['full', 'freeze_half']):
            nm = f"lr{'penuh' if lr == b['lr'] else 'setengah'}_{strat}"
            cfgs.append(make('F7', b['backbone'], T2, tag=f'tahap2saja_{nm}', seed=seed, varian='tahap2_saja_grid', lr=lr,
                             strategy=strat, loss='ce', batch_size=16, **arch))
            cfgs.append(make('F7', b['backbone'], satu_val, tag=f'satutahap_{nm}', seed=seed, varian='satu_tahap_grid', lr=lr,
                             strategy=strat, loss=b['loss'], batch_size=16, **arch))
        c2 = [c for c in cfgs2 if c['seed'] == seed][0]
        s1 = make('F7', b['backbone'], T1, tag='tahap1_cls', seed=seed, save_model=True, varian='tahap1_cls', head='cls',
                  strategy='full', loss=b['loss'], lr=b['lr'], dropout=b['dropout'], batch_size=b['batch_efektif'])
        cfgs.append(s1)
        cfgs.append(make('F7', b['backbone'], {**T2, 'train_files': c2['train_files']}, tag='duatahap_cls', seed=seed,
                         init_from=s1['out_dir'], varian='dua_tahap_cls', head='cls', strategy=c2['strategy'], loss=c2['loss'],
                         lr=c2['lr'], dropout=b['dropout'], batch_size=16, rehearsal=c2.get('rehearsal'), lr_tahap2=c2['lr']))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--semua-backbone', action='store_true')
    ap.add_argument('--rencana', action='store_true')
    ap.add_argument('--backbone', nargs='*', default=None, help='batasi ke backbone tertentu (satu per proses untuk menghindari OOM)')
    a = ap.parse_args()
    print(f'GPU VRAM terdeteksi: {vram_gb():.1f} GB')
    fns = {1: fase1, 2: lambda: fase2(a.semua_backbone), 3: lambda: fase3(a.semua_backbone),
           4: lambda: fase4(a.semua_backbone), 5: lambda: fase5(a.semua_backbone), 6: lambda: fase6(a.semua_backbone),
           7: lambda: fase7(a.semua_backbone)}
    for f in a.fase:
        cfgs = fns[f]()
        if a.backbone:
            cfgs = [c for c in cfgs if c['backbone'] in a.backbone]
        print(f'\n=== FASE {f} ===')
        if not cfgs:
            print('  (menunggu hasil fase sebelumnya)'); continue
        execute(cfgs, a.rencana)
        # fase 5/6: pastikan tahap 1 dijalankan lebih dulu (execute berurutan sesuai daftar)
    kumpulkan()
    print('\nRingkasan tersimpan di hasil/semua_run.csv. Jalankan: python rekap_hasil.py')


if __name__ == '__main__':
    main()
