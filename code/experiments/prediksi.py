"""Contoh pemakaian model final (untuk modul chatbot).
  python prediksi.py "aku takut banget besok sidang"
"""
import sys, os, json, torch
from modeling import EmotionClassifier, load_tokenizer, BACKBONES
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'hasil', 'model_final')
info = json.load(open(os.path.join(D, 'info.json')))
tok = load_tokenizer(BACKBONES[info['backbone']])
model = EmotionClassifier(BACKBONES[info['backbone']], info['head'], info['dropout'])
model.load_state_dict(torch.load(os.path.join(D, 'model.pt'), map_location='cpu', weights_only=True), strict=True)
model.eval()

def petakan_emosi(teks):
    enc = tok([teks], truncation=True, max_length=info['max_len'], return_tensors='pt')
    enc = {k: v for k, v in enc.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
    with torch.no_grad():
        p = torch.softmax(model(**enc) / info['temperature'], -1)[0]
    probs = {l: round(float(x), 4) for l, x in zip(info['labels'], p)}
    return max(probs, key=probs.get), probs

if __name__ == '__main__':
    for t in sys.argv[1:] or ['aku takut banget besok sidang', 'hari ini aku senang sekali']:
        print(t, '->', petakan_emosi(t))
