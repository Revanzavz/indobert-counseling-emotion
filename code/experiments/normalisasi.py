"""Normalisasi teks informal bahasa Indonesia berbasis kamus.

Kamus: Colloquial Indonesian Lexicon (Salsabila et al., 2018), file
colloquial-indonesian-lexicon.csv (kolom slang, formal). Contoh: "gak" -> "tidak", "bgt" -> "banget".
"""
import re, csv


class Normalizer:
    def __init__(self, path):
        self.map = {}
        with open(path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                s, f = r['slang'].strip().lower(), r['formal'].strip().lower()
                if s and f and s != f and s not in self.map:
                    self.map[s] = f
        self.tok = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __call__(self, text):
        t = str(text).lower()
        t = re.sub(r'(.)\1{2,}', r'\1\1', t)          # "bangeeet" -> "bangeet"
        return ' '.join(self.map.get(w, w) for w in t.split())
