"""Tampilkan progres eksperimen: jumlah run selesai, run yang sedang berjalan, epoch terakhir, dan GPU."""
import os, glob, json, time, subprocess, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    now = datetime.datetime.now()
    print(f'STATUS EKSPERIMEN  ({now:%d-%m-%Y %H:%M:%S})\n')
    dirs = sorted(glob.glob(os.path.join(RUNS, '*')), key=os.path.getmtime)
    done = [d for d in dirs if os.path.exists(os.path.join(d, 'metrics.json'))]
    fail = [d for d in dirs if os.path.exists(os.path.join(d, 'status.json'))]
    per_fase = {}
    for d in done:
        f = os.path.basename(d).split('_')[0]; per_fase[f] = per_fase.get(f, 0) + 1
    print('Run selesai :', len(done), ' ', dict(sorted(per_fase.items())))
    if any(os.path.basename(d).startswith('F7_') for d in dirs):
        print(f"Fase 7      : {per_fase.get('F7', 0)} dari 30 run selesai" + ('  (SELESAI, lihat hasil\\rekap_fase7_pembanding_adil.md)' if os.path.exists(os.path.join(HERE, 'hasil', 'rekap_fase7_pembanding_adil.md')) else ''))
    print('Run gagal/OOM:', len(fail), [os.path.basename(d)[:60] for d in fail][-3:])
    running = [d for d in dirs if not os.path.exists(os.path.join(d, 'metrics.json')) and d not in fail]
    if running:
        d = running[-1]; log = os.path.join(d, 'log.txt')
        age = time.time() - os.path.getmtime(log) if os.path.exists(log) else 0
        print('\nSedang berjalan:', os.path.basename(d))
        if os.path.exists(log):
            ep = [l for l in open(log, encoding='utf-8', errors='ignore') if l.startswith('{"epoch"')]
            print('  epoch selesai:', len(ep), '| log terakhir diperbarui', int(age // 60), 'menit lalu')
            if ep:
                e = json.loads(ep[-1]); print(f"  epoch {e['epoch']}: val_loss {e['val_loss']:.4f}  val_f1_weighted {e['val_f1_weighted']:.4f}")
    if done:
        m = json.load(open(os.path.join(done[-1], 'metrics.json')))
        print('\nRun terakhir selesai:', os.path.basename(done[-1]))
        print('  test:', {k: round(v['f1_weighted'], 4) for k, v in m['test'].items()}, f"| {m['durasi_detik']/60:.1f} menit")
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        print('\nGPU (pemakaian, memori terpakai, total, suhu):', g)
        print('  Pemakaian GPU tinggi (>50%) berarti training sedang berjalan.')
    except Exception:
        pass
    print('\nHalaman ini diperbarui tiap 60 detik. Tutup jendela ini kapan saja, training tidak terpengaruh.')
    time.sleep(60)
