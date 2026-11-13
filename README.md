# Text-Based User Emotion Mapping Using IndoBERT for Virtual Counseling Chatbots

Code, data indices, and results for the paper *Text-Based User Emotion Mapping Using IndoBERT for Virtual Counseling Chatbots* (Naufal Arrafi Revanza, Shaufiah, Rifki Wijaya; School of Computing, Telkom University).

The project builds a five-class Indonesian emotion classifier (anger, fear, happy, sadness, neutral) for counseling-style text, compares two-stage and single-stage fine-tuning of IndoBERT variants, and integrates the model into a research prototype Telegram chatbot. The chatbot uses a crisis detector and sends only a fixed-schema conversation map to Gemini, never the user's text. **This is a research prototype, not a mental health or crisis service.**

## Repository layout

| Folder | Contents |
|---|---|
| `code/experiments/` | Training and evaluation code for the 204 runs (`train.py`, `modeling.py`, `jalankan_eksperimen.py`), crisis detector training (`latih_krisis.py`), Gemini comparison analysis, `requirements.txt` |
| `code/data_pipeline/` | Dataset construction: EmpatheticDialogues extraction and NLLB-200 translation, DailyDialog neutral selection, EmoT preparation, both crisis datasets, kit for a native test set |
| `code/chatbot/` | Telegram bot, crisis gate, emotion model wrapper, dialogue-act rules, conversation-map builder with privacy check, functional, crisis, and privacy tests. Copy `.env.contoh` to `.env` and fill in your own keys |
| `data/label_map_*.csv` | Full mapping of the 32 EmpatheticDialogues labels (24 mapped, 8 dropped) and the Riccosan labels |
| `data/stats/` | Item counts after every filter (EmpatheticDialogues, DailyDialog, crisis data, EmoT) and the TF-IDF baselines |
| `data/splits/` | Train, validation, test, and rehearsal splits as item IDs, labels, sources, and SHA-256 hashes of the texts |
| `data/counseling_translated/` | Indonesian translations of the counseling-domain data (CC BY-NC-SA 4.0, see its LICENSE) |
| `results/semua_run.csv` | Ledger of all 204 runs: phase, configuration, seed, best epoch, duration, temperature, validation and test scores |
| `results/runs/<run_id>/` | Config, full metrics (including ECE before and after calibration), and per-item predictions for every run and seed |
| `results/` | Phase summaries, matched-budget comparison, Gemini comparison (per-item predictions, per-class tests), EmoT and crisis predictions, chatbot test reports |
| `results/models/` | Selection info, temperature, and thresholds of the deployed emotion and crisis models |
| `prompts/` | Exact Gemini prompts: zero-shot classifier and response-module system instruction |
| `analysis_revision/` | Scripts for the analyses added after review: item-bootstrap with seed blocks, Gemini intervals with Holm correction, three-way crisis outcomes with Wilson intervals, duplicate checks, style-only baseline, token lengths |

## Data sources and licenses

Tweet texts are not redistributed. Splits list hashes so that the same items can be matched after downloading the original data.

| Data | Source | License |
|---|---|---|
| Stage-1 tweets | Riccosan et al., *Data in Brief* 43 (2022) | CC BY-NC-ND 4.0 (hashes only) |
| Counseling emotions | EmpatheticDialogues (Rashkin et al., 2019) | CC BY-NC 4.0 |
| Counseling neutral | DailyDialog (Li et al., 2017), ConvLab copy on HuggingFace | CC BY-NC-SA 4.0 |
| Native crisis tweets | Nataputra (2023), github.com/apricitea/text-suicide-ideation-detection | hashes only |
| Translated crisis posts | Sentiment Analysis for Mental Health (Kaggle, Sarkar) | hashes only |
| Evaluation | EmoT (Saputri et al., 2018) via IndoNLU | hashes only |

The 50 crisis test messages (`code/chatbot/uji_krisis.csv`) and the 30 functional cases (`code/chatbot/uji_kasus.csv`) were written by the authors and are included in full.

## Reproducing the experiments

1. Install PyTorch with CUDA, then `pip install -r code/experiments/requirements.txt`.
2. Download the original datasets listed above and rebuild the splits with the scripts in `code/data_pipeline/` (all splits use seed 42; check them against the hashes in `data/splits/`).
3. Run `python jalankan_eksperimen.py` in `code/experiments/` (phases 1 to 6), then the phase 7 script. Finished runs are skipped, so the runner can be stopped and resumed.
4. Train the crisis detectors with `latih_krisis.py`.
5. Model weights are not included because of their size and the non-commercial licenses of the training data.

The original experiments ran on an NVIDIA RTX 3050 laptop GPU (4 GB); the 204 runs took about 46 hours.

## Running the chatbot

Copy `code/chatbot/.env.contoh` to `.env`, set `GEMINI_API_KEY` and `TELEGRAM_BOT_TOKEN`, place the trained models where `konfigurasi.py` expects them, and run `jalankan_telegram.bat` (or `python server_telegram.py`). The bot keeps per-chat history in memory only and logs no message text unless the optional `PANTAU=1` monitoring mode is enabled for testing.

## License

Code: MIT (see `LICENSE`). Data derived from third-party datasets keeps the license of its source.
