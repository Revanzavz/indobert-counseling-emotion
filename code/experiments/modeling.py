"""Komponen model: backbone IndoBERT + variasi classification head + strategi fine-tuning."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizerFast

LABELS = ['anger', 'fear', 'happy', 'sadness', 'neutral']
L2I = {l: i for i, l in enumerate(LABELS)}

# ---------------------------------------------------------------- backbone & tokenizer
BACKBONES = {
    # nama pendek            : id HuggingFace
    'indobert-base-p1':        'indobenchmark/indobert-base-p1',
    'indobert-base-p2':        'indobenchmark/indobert-base-p2',
    'indobert-large-p1':       'indobenchmark/indobert-large-p1',
    'indobert-large-p2':       'indobenchmark/indobert-large-p2',
    'indobert-lite-base-p1':   'indobenchmark/indobert-lite-base-p1',
    'indobert-lite-base-p2':   'indobenchmark/indobert-lite-base-p2',
    'indobert-lite-large-p1':  'indobenchmark/indobert-lite-large-p1',
    'indobert-lite-large-p2':  'indobenchmark/indobert-lite-large-p2',
    'indolem-indobert-base':   'indolem/indobert-base-uncased',
    'indobertweet-base':       'indolem/indobertweet-base-uncased',
}
# Opsional: ganti daftar backbone lewat file JSON (mis. untuk path lokal atau uji cepat).
import os as _os, json as _json
if _os.environ.get('TA_BACKBONES'):
    BACKBONES = _json.load(open(_os.environ['TA_BACKBONES']))


def load_tokenizer(name_or_path):
    cfg = AutoConfig.from_pretrained(name_or_path)
    # Model IndoBERT-lite berarsitektur ALBERT tetapi memakai vocab WordPiece (vocab.txt),
    # sehingga AutoTokenizer akan salah memilih AlbertTokenizer. Paksa BertTokenizerFast.
    if cfg.model_type == 'albert':
        return BertTokenizerFast.from_pretrained(name_or_path, do_lower_case=True)
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception:
        return BertTokenizerFast.from_pretrained(name_or_path, do_lower_case=True)


# ---------------------------------------------------------------- heads
class ClsHead(nn.Module):
    """Rancangan proposal: representasi [CLS] layer terakhir -> dropout -> dense."""
    def __init__(self, h, n, p):
        super().__init__(); self.drop = nn.Dropout(p); self.out = nn.Linear(h, n)
    def forward(self, hs, mask):
        return self.out(self.drop(hs[:, 0]))


def masked_mean(hs, mask):
    m = mask.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)


class MeanHead(nn.Module):
    def __init__(self, h, n, p):
        super().__init__(); self.drop = nn.Dropout(p); self.out = nn.Linear(h, n)
    def forward(self, hs, mask):
        return self.out(self.drop(masked_mean(hs, mask)))


class ClsMeanHead(nn.Module):
    def __init__(self, h, n, p):
        super().__init__(); self.drop = nn.Dropout(p); self.out = nn.Linear(2 * h, n)
    def forward(self, hs, mask):
        return self.out(self.drop(torch.cat([hs[:, 0], masked_mean(hs, mask)], -1)))


class BiLSTMHead(nn.Module):
    """IndoBERT + BiLSTM (mengikuti pendekatan hybrid Ahmadian et al.), pooling mean+max."""
    def __init__(self, h, n, p, lstm_h=256, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(h, lstm_h, batch_first=True, bidirectional=True)
        self.attention = attention
        if attention:
            self.att = nn.Sequential(nn.Linear(2 * lstm_h, 128), nn.Tanh(), nn.Linear(128, 1))
            self.out = nn.Linear(2 * lstm_h, n)
        else:
            self.out = nn.Linear(4 * lstm_h, n)
        self.drop = nn.Dropout(p)

    def forward(self, hs, mask):
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(hs.float(), lengths, batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=hs.size(1))
        m = mask.unsqueeze(-1).bool()
        if self.attention:
            s = self.att(o).masked_fill(~m, -1e4)
            a = torch.softmax(s, 1)
            z = (a * o).sum(1)
        else:
            mean = (o * m).sum(1) / m.sum(1).clamp(min=1)
            mx = o.masked_fill(~m, -1e4).max(1).values
            z = torch.cat([mean, mx], -1)
        return self.out(self.drop(z))


HEADS = {
    'cls': lambda h, n, p: ClsHead(h, n, p),
    'mean': lambda h, n, p: MeanHead(h, n, p),
    'cls_mean': lambda h, n, p: ClsMeanHead(h, n, p),
    'bilstm': lambda h, n, p: BiLSTMHead(h, n, p, attention=False),
    'bilstm_attn': lambda h, n, p: BiLSTMHead(h, n, p, attention=True),
}


class EmotionClassifier(nn.Module):
    def __init__(self, backbone_path, head='cls', dropout=0.1, n_labels=len(LABELS), grad_ckpt=False):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(backbone_path, add_pooling_layer=False)
        except TypeError:
            self.encoder = AutoModel.from_pretrained(backbone_path)
        if grad_ckpt:
            # use_reentrant=False: checkpointing reentrant memutus gradien bila embedding dibekukan
            # (freeze_half / gradual_unfreeze), sehingga layer encoder yang dibuka tidak ikut terlatih.
            try:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
                self.encoder.enable_input_require_grads()
        self.model_type = self.encoder.config.model_type
        self.head_name = head
        self.head = HEADS[head](self.encoder.config.hidden_size, n_labels, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw['token_type_ids'] = token_type_ids
        hs = self.encoder(**kw).last_hidden_state
        return self.head(hs, attention_mask)


# ---------------------------------------------------------------- kelompok layer (untuk LLRD / freezing)
def layer_groups(model):
    """Mengembalikan daftar kelompok parameter dari bawah ke atas:
    [embeddings, layer_0, ..., layer_{L-1}] untuk BERT.
    ALBERT (IndoBERT-lite) memakai satu blok encoder yang dibagi bersama semua layer, sehingga
    kelompoknya hanya [embeddings, encoder_bersama]."""
    enc = model.encoder
    if model.model_type == 'albert':
        emb = list(enc.embeddings.parameters())
        rest = [p for n, p in enc.named_parameters() if not n.startswith('embeddings.')]
        return [emb, rest]
    groups = [list(enc.embeddings.parameters())]
    for layer in enc.encoder.layer:
        groups.append(list(layer.parameters()))
    covered = {id(p) for g in groups for p in g}
    extra = [p for p in enc.parameters() if id(p) not in covered]
    if extra:
        groups[-1] += extra
    return groups


def build_param_groups(model, strategy, lr, head_lr, weight_decay, llrd_decay=0.9):
    """Parameter group AdamW sesuai strategi fine-tuning."""
    groups = layer_groups(model)
    head_params = list(model.head.parameters())
    def wd_split(params, lr_):
        return [{'params': [p for p in params if p.ndim > 1], 'lr': lr_, 'weight_decay': weight_decay},
                {'params': [p for p in params if p.ndim <= 1], 'lr': lr_, 'weight_decay': 0.0}]
    pg = []
    if strategy == 'llrd':
        n = len(groups)
        for i, g in enumerate(groups):           # layer paling atas mendapat lr penuh
            pg += wd_split(g, lr * (llrd_decay ** (n - 1 - i)))
    else:
        pg += wd_split([p for g in groups for p in g], lr)
    pg += wd_split(head_params, head_lr)
    return [g for g in pg if g['params']]


def apply_freeze(model, strategy, epoch, n_epochs):
    """Mengatur requires_grad sesuai strategi.
    full / llrd       : semua parameter dilatih.
    freeze_half       : embeddings + setengah layer bawah dibekukan selama pelatihan.
    gradual_unfreeze  : (ULMFiT) epoch pertama hanya head, lalu tiap epoch membuka satu blok
                        berikutnya dari atas ke bawah (maksimal 4 blok).
    Mengembalikan jumlah blok untuk gradual_unfreeze (0 untuk strategi lain)."""
    groups = layer_groups(model)
    for g in groups:
        for p in g:
            p.requires_grad = True
    if strategy == 'freeze_half':
        k = max(1, len(groups) // 2)
        for g in groups[:k]:
            for p in g:
                p.requires_grad = False
    elif strategy == 'gradual_unfreeze':
        n = len(groups)
        # bagi kelompok menjadi 4 blok (atau sebanyak kelompok bila < 4)
        n_blocks = min(4, n)
        top_down = groups[::-1]                          # dari layer teratas ke embeddings
        cuts = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
        blocks = [top_down[cuts[b]:cuts[b + 1]] for b in range(n_blocks)]
        open_blocks = min(epoch, n_blocks)       # epoch 0 -> hanya head
        for b, blk in enumerate(blocks):
            for g in blk:
                for p in g:
                    p.requires_grad = b < open_blocks
    for p in model.head.parameters():
        p.requires_grad = True
    return min(4, len(groups)) if strategy == 'gradual_unfreeze' else 0


# ---------------------------------------------------------------- loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.register_buffer('w', weight if weight is not None else None)
    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), -1)
        ce = F.nll_loss(logp, y, weight=self.w, reduction='none')
        pt = logp.gather(1, y.unsqueeze(1)).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss(name, class_weights):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == 'focal':
        return FocalLoss(2.0, class_weights)
    if name == 'label_smoothing':
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    raise ValueError(name)
