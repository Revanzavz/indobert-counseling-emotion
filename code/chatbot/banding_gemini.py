"""Pembanding: Gemini sebagai pengklasifikasi emosi tanpa pelatihan (zero-shot) pada data uji yang sama
dengan model IndoBERT final.

  python banding_gemini.py              # data uji konseling (400 kalimat)
  python banding_gemini.py --tweet      # tambah data uji tweet Riccosan (700 kalimat)

Kalimat dikirim per kelompok supaya hemat kuota. Hasil tiap kelompok langsung disimpan, sehingga bila
kuota habis skrip cukup dijalankan lagi dan akan melanjutkan dari kelompok yang belum selesai.
Keluaran: ../Eksperimen/eksperimen/hasil/banding_gemini_<set>.csv, banding_gemini.json, banding_gemini.md
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from konfigurasi import GEMINI_API_KEY, GEMINI_MODEL, DIR_EKSPERIMEN

LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
UKURAN_KELOMPOK = 50          # 300 kalimat pertama dikirim per 25; sisanya per 50 karena kuota harian sangat kecil
DATA = os.path.join(DIR_EKSPERIMEN, '..', 'data_final')
HASIL = os.path.join(DIR_EKSPERIMEN, 'hasil')
RUNS = os.path.join(DIR_EKSPERIMEN, 'runs')

INSTRUKSI = """Kamu adalah pengklasifikasi emosi untuk teks berbahasa Indonesia.
Tentukan satu label emosi untuk setiap teks. Label yang tersedia:
- anger: marah, kesal, jengkel, iri
- fear: takut, cemas, khawatir, gugup
- happy: senang, gembira, bangga, bersyukur, penuh harap, sayang
- sadness: sedih, kecewa, kesepian, malu, merasa bersalah, rindu masa lalu
- neutral: tidak menunjukkan emosi tertentu, hanya pernyataan atau percakapan biasa
Jawab hanya dengan JSON berbentuk daftar objek {"no": nomor teks, "label": salah satu label di atas},
satu objek untuk setiap teks, dengan urutan yang sama."""


class KuotaHabis(Exception):
    pass


class Diblokir(Exception):
    """Gemini tidak memberi jawaban karena filter keamanan (biasanya kalimat tentang kematian atau kekerasan)."""
    pass


KEAMANAN = [{'category': c, 'threshold': 'BLOCK_NONE'} for c in
            ['HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH', 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
             'HARM_CATEGORY_DANGEROUS_CONTENT']]


def panggil(teks_list):
    isi = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(teks_list))
    badan = {'systemInstruction': {'parts': [{'text': INSTRUKSI}]},
             'contents': [{'role': 'user', 'parts': [{'text': isi}]}],
             'safetySettings': KEAMANAN,
             'generationConfig': {'temperature': 0, 'maxOutputTokens': 4096, 'responseMimeType': 'application/json',
                                  'thinkingConfig': {'thinkingBudget': 0}}}
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
    for coba in range(6):
        req = urllib.request.Request(url, data=json.dumps(badan).encode('utf-8'),
                                     headers={'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            kand = d.get('candidates') or []
            if not kand or 'content' not in kand[0] or not kand[0]['content'].get('parts'):
                alasan = (d.get('promptFeedback') or {}).get('blockReason') or (kand[0].get('finishReason') if kand else None)
                raise Diblokir(alasan or 'tanpa jawaban')
            teks = ''.join(p.get('text', '') for p in d['candidates'][0]['content']['parts'])
            out = json.loads(teks)
            lab = {int(o['no']): str(o['label']).strip().lower() for o in out}
            hasil = [lab.get(i + 1) for i in range(len(teks_list))]
            if any(h not in LABELS for h in hasil):
                raise ValueError(f'jawaban tidak lengkap atau label tidak dikenal: {hasil}')
            return hasil
        except urllib.error.HTTPError as e:
            pesan = e.read().decode('utf-8', 'ignore')
            if e.code == 400 and 'thinking' in pesan.lower() and 'thinkingConfig' in badan['generationConfig']:
                del badan['generationConfig']['thinkingConfig']; continue
            if e.code == 429:
                if 'PerDay' in pesan or 'per day' in pesan.lower():
                    raise KuotaHabis('kuota harian Gemini habis')
                m = re.search(r'retry in ([0-9.]+)s', pesan)
                tunggu = min(float(m.group(1)) + 2 if m else 30, 120)
                print(f'   batas per menit, menunggu {tunggu:.0f} detik...', flush=True); time.sleep(tunggu); continue
            if e.code in (500, 503):
                time.sleep(15); continue
            raise RuntimeError(f'HTTP {e.code}: {pesan[:300]}')
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f'   jawaban tidak valid ({e}), mengulang...', flush=True); time.sleep(3)
    raise KuotaHabis('gagal berulang kali (kemungkinan kuota), jalankan lagi nanti')


def panggil_aman(teks_list):
    """Bila satu kelompok diblokir, kelompok dibelah dua sampai kalimat penyebabnya ketemu.
    Kalimat yang tetap diblokir sendirian diberi label 'diblokir' dan dihitung sebagai jawaban salah."""
    try:
        return panggil(teks_list)
    except Diblokir as e:
        if len(teks_list) == 1:
            print(f'   1 kalimat diblokir Gemini ({e}): {teks_list[0][:70]}', flush=True)
            return ['diblokir']
        print(f'   kelompok {len(teks_list)} kalimat diblokir ({e}), dibelah dua...', flush=True)
        h = len(teks_list) // 2
        kiri = panggil_aman(teks_list[:h]); time.sleep(4)
        return kiri + panggil_aman(teks_list[h:])


def klasifikasi(nama, berkas):
    d = pd.read_csv(os.path.join(DATA, berkas))
    out = os.path.join(HASIL, f'banding_gemini_{nama}.csv')
    siap = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=['id', 'pred_gemini', 'detik'])
    sudah = set(siap.id)
    sisa = d[~d.id.isin(sudah)]
    print(f'[{nama}] {len(d)} kalimat, {len(sudah)} sudah selesai, {len(sisa)} akan diklasifikasi', flush=True)
    for k in range(0, len(sisa), UKURAN_KELOMPOK):
        b = sisa.iloc[k:k + UKURAN_KELOMPOK]
        t0 = time.time()
        pred = panggil_aman(b.text.astype(str).tolist())
        dt = (time.time() - t0) / len(b)
        siap = pd.concat([siap, pd.DataFrame({'id': b.id.values, 'pred_gemini': pred, 'detik': dt})], ignore_index=True)
        siap.to_csv(out, index=False)
        print(f'   kelompok {k // UKURAN_KELOMPOK + 1}: {len(b)} kalimat, {time.time() - t0:.1f} dtk', flush=True)
        time.sleep(7)                                   # jeda supaya tidak melewati batas permintaan per menit
    return d.merge(siap, on='id', how='left')


def skor(y, p):
    return {'accuracy': accuracy_score(y, p), 'f1_macro': f1_score(y, p, average='macro', labels=LABELS, zero_division=0),
            'f1_weighted': f1_score(y, p, average='weighted', labels=LABELS, zero_division=0),
            'per_kelas': dict(zip(LABELS, f1_score(y, p, average=None, labels=LABELS, zero_division=0).round(4).tolist())),
            'confusion_matrix': confusion_matrix(y, p, labels=LABELS).tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tweet', action='store_true'); a = ap.parse_args()
    if not GEMINI_API_KEY:
        sys.exit('GEMINI_API_KEY belum diisi di .env')
    info = json.load(open(os.path.join(HASIL, 'model_final', 'info.json')))
    sets = [('konseling', 'tahap2_test.csv', 'tahap2_test')] + ([('tweet', 'tahap1_test.csv', 'tahap1_test')] if a.tweet else [])
    ringkas, belum = {'model_gemini': GEMINI_MODEL, 'model_indobert': info['run']}, False
    for nama, berkas, kunci in sets:
        try:
            d = klasifikasi(nama, berkas)
        except KuotaHabis as e:
            print(f'\nBERHENTI: {e}. Jalankan lagi skrip ini nanti, hasil yang sudah ada tidak diulang.'); belum = True; break
        ib = pd.read_csv(os.path.join(RUNS, info['run'], 'prediksi', f'{kunci}.csv'))[['id', 'pred']].rename(columns={'pred': 'pred_indobert'})
        d = d.merge(ib, on='id')
        g, i = skor(d.label, d.pred_gemini), skor(d.label, d.pred_indobert)
        g['detik_per_kalimat'] = float(d.detik.mean())
        g['diblokir'] = int((d.pred_gemini == 'diblokir').sum())
        hanya_g = int(((d.pred_gemini == d.label) & (d.pred_indobert != d.label)).sum())
        hanya_i = int(((d.pred_gemini != d.label) & (d.pred_indobert == d.label)).sum())
        try:
            from scipy.stats import binomtest
            p = float(binomtest(hanya_g, hanya_g + hanya_i).pvalue) if hanya_g + hanya_i else 1.0
        except Exception:
            p = None
        ringkas[nama] = {'n': len(d), 'gemini': g, 'indobert': i,
                         'mcnemar': {'benar_hanya_gemini': hanya_g, 'benar_hanya_indobert': hanya_i, 'p': p},
                         'kesepakatan_prediksi': float((d.pred_gemini == d.pred_indobert).mean())}
        d.to_csv(os.path.join(HASIL, f'banding_gemini_{nama}_lengkap.csv'), index=False)
    if belum and len(ringkas) == 2:
        return
    json.dump(ringkas, open(os.path.join(HASIL, 'banding_gemini.json'), 'w'), indent=1)
    with open(os.path.join(HASIL, 'banding_gemini.md'), 'w', encoding='utf-8') as f:
        for nama in [s[0] for s in sets if s[0] in ringkas]:
            r = ringkas[nama]; g, i = r['gemini'], r['indobert']
            f.write(f"## Data uji {nama} ({r['n']} kalimat)\n\n| Metrik | {GEMINI_MODEL} (zero-shot) | IndoBERTweet dua tahap |\n|---|---|---|\n")
            for k, n in [('accuracy', 'Accuracy'), ('f1_macro', 'Macro F1'), ('f1_weighted', 'Weighted F1')]:
                f.write(f'| {n} | {g[k]:.4f} | {i[k]:.4f} |\n')
            for l in LABELS:
                f.write(f"| F1 {l} | {g['per_kelas'][l]:.4f} | {i['per_kelas'][l]:.4f} |\n")
            m = r['mcnemar']
            f.write(f"\nUji McNemar: benar hanya Gemini {m['benar_hanya_gemini']}, benar hanya IndoBERTweet {m['benar_hanya_indobert']}, "
                    f"p = {m['p']:.4f}. Kesepakatan prediksi kedua model {r['kesepakatan_prediksi']:.1%}. "
                    f"Waktu Gemini {g['detik_per_kalimat']:.2f} detik per kalimat (dikirim per 25 sampai 50 kalimat). "
                    f"Kalimat yang diblokir filter keamanan Gemini (dihitung salah): {g['diblokir']}.\n\n")
    print('\n' + open(os.path.join(HASIL, 'banding_gemini.md'), encoding='utf-8').read())


if __name__ == '__main__':
    main()
