import sys, json
info = {'python': sys.executable, 'versi': sys.version.split()[0]}
try:
    import torch
    info['torch'] = torch.__version__
    info['cuda'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info['gpu'] = p.name; info['vram_gb'] = round(p.total_memory / 1e9, 2)
except Exception as e:
    info['torch'] = 'TIDAK ADA (' + type(e).__name__ + ')'
for m in ['transformers', 'sklearn', 'pandas', 'matplotlib', 'openpyxl', 'tabulate', 'bitsandbytes']:
    try:
        mod = __import__(m); info[m] = getattr(mod, '__version__', 'ada')
    except Exception:
        info[m] = None
print(json.dumps(info))
