"""Membuat lembar pelabelan untuk dua penilai dari data_asli.csv (urutan diacak, tanpa kolom petunjuk kelas)."""
import pandas as pd
d = pd.read_csv('data_asli.csv').dropna(subset=['teks'])
d = d[~d.teks.astype(str).str.startswith('contoh:')]
assert d.id.is_unique, 'kolom id harus unik'
for nama, seed in [('A', 1), ('B', 2)]:
    s = d[['id', 'teks']].sample(frac=1, random_state=seed)
    s['label'] = ''; s['catatan'] = ''
    s.to_csv(f'lembar_penilai_{nama}.csv', index=False, encoding='utf-8-sig')
print(f'{len(d)} kalimat. Dibuat: lembar_penilai_A.csv dan lembar_penilai_B.csv')
