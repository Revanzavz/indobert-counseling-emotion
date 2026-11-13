"""Modul klasifikasi emosi: memuat model final IndoBERT dari folder eksperimen (dimuat sekali)."""
import os, sys
from konfigurasi import DIR_EKSPERIMEN

_petakan = None


def petakan_emosi(teks):
    """Mengembalikan (label, probs) dengan probs = {emosi: probabilitas terkalibrasi}."""
    global _petakan
    if _petakan is None:
        os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
        d = os.path.abspath(DIR_EKSPERIMEN)
        if d not in sys.path:
            sys.path.insert(0, d)
        from prediksi import petakan_emosi as f      # memuat hasil/model_final
        _petakan = f
    return _petakan(teks)
