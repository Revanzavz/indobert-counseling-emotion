"""Deteksi krisis berbasis model: IndoBERTweet biner (krisis / bukan) yang dilatih dari kelas Suicidal
dataset Kaggle "Sentiment Analysis for Mental Health" (diterjemahkan dengan NLLB-200).

Model mengeluarkan probabilitas krisis yang sudah dikalibrasi. Dua ambang dipilih pada data validasi:
  p >= ambang_krisis : pesan dialihkan ke pesan rujukan, tidak diteruskan ke model emosi maupun Gemini
  p >= ambang_hati   : pesan diproses biasa, tetapi Gemini diberi penanda kehati-hatian
Model disimpan di Eksperimen/eksperimen/hasil/model_krisis_id (dibuat oleh latih_krisis_data_id.bat)."""
import json, os, sys
from konfigurasi import DIR_EKSPERIMEN

PESAN_KRISIS = (
    "Terima kasih sudah mau cerita. Aku khawatir dengan keselamatanmu, dan apa yang kamu rasakan "
    "sekarang penting untuk ditangani bersama orang yang terlatih.\n\n"
    "Kamu bisa menghubungi layanan berikut:\n"
    "- Layanan kesehatan jiwa Kementerian Kesehatan: telepon 119 lalu tekan 8, atau buka healing119.id\n"
    "- Keadaan darurat: telepon 112 atau datang ke IGD rumah sakit terdekat\n\n"
    "Kalau bisa, hubungi juga orang yang kamu percaya dan minta ditemani sekarang. "
    "Aku hanya program komputer dan tidak bisa menggantikan bantuan profesional, "
    "tapi kamu tidak harus menghadapi ini sendirian."
)

# model_krisis_id: dilatih dengan tweet berbahasa Indonesia asli (apricitea); model lama (Kaggle terjemahan) di model_krisis
DIR_MODEL = os.path.join(os.path.abspath(DIR_EKSPERIMEN), 'hasil', os.environ.get('MODEL_KRISIS', 'model_krisis_id'))
_m = None


def _muat():
    global _m
    if _m is not None:
        return _m
    info_p = os.path.join(DIR_MODEL, 'info.json')
    if not os.path.exists(info_p):
        raise RuntimeError('Model deteksi krisis belum dilatih. Jalankan Dataset/Krisis_Kaggle/jalankan_krisis.bat terlebih dahulu.')
    import torch
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    if os.path.abspath(DIR_EKSPERIMEN) not in sys.path:
        sys.path.insert(0, os.path.abspath(DIR_EKSPERIMEN))
    from modeling import EmotionClassifier, load_tokenizer, BACKBONES
    info = json.load(open(info_p))
    tok = load_tokenizer(BACKBONES[info['backbone']])
    model = EmotionClassifier(BACKBONES[info['backbone']], info['head'], info['dropout'], n_labels=2)
    model.load_state_dict(torch.load(os.path.join(DIR_MODEL, 'model.pt'), map_location='cpu', weights_only=True), strict=True)
    model.eval()
    _m = (tok, model, info, torch)
    return _m


def nilai_risiko(teks):
    """Probabilitas terkalibrasi bahwa pesan berindikasi krisis."""
    tok, model, info, torch = _muat()
    enc = tok([teks], truncation=True, max_length=info['max_len'], return_tensors='pt')
    enc = {k: v for k, v in enc.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
    with torch.no_grad():
        p = torch.softmax(model(**enc) / info['temperature'], -1)[0, 1].item()
    return p


def periksa(teks):
    """Mengembalikan (status, p) dengan status 'krisis', 'hati_hati', atau None."""
    _, _, info, _ = _muat()
    p = nilai_risiko(teks)
    if p >= info['ambang_krisis']:
        return 'krisis', p
    if p >= info['ambang_hati']:
        return 'hati_hati', p
    return None, p
