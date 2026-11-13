"""Coba model final secara interaktif: ketik kalimat, model menampilkan emosi dan probabilitasnya.
Jalankan lewat coba_model.bat (double-click). Ketik 'keluar' untuk berhenti."""
import sys, os
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
print('Memuat model final, tunggu sebentar...', flush=True)
from prediksi import petakan_emosi, info

def tampil(teks):
    label, probs = petakan_emosi(teks)
    print(f'  -> Emosi: {label.upper()}')
    for l, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f'     {l:8s} {p*100:5.1f}%  ' + '#' * int(round(p * 30)))

if __name__ == '__main__':
    if len(sys.argv) > 1:                       # mode uji: kalimat dari argumen
        for t in sys.argv[1:]:
            print(t); tampil(t)
        sys.exit()
    print(f"Model: {info['backbone']} + {info['head']} ({info['skenario']})")
    print("Ketik kalimat lalu tekan Enter. Ketik 'keluar' untuk berhenti.\n")
    while True:
        try:
            t = input('Kalimat: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if t.lower() in ('keluar', 'exit', 'q'):
            break
        if t:
            tampil(t); print()
