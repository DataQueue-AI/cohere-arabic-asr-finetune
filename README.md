# Cohere Arabic ASR Fine-tuning

Full fine-tune of [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) (2.07B, Conformer encoder-decoder) on Arabic speech data.

## Experiments

All experiments, resulting checkpoints, WER/CER, and their GCS path are tracked here: [STT-Models - ML - Confluence](https://dataqueue.atlassian.net/wiki/spaces/ML/database/2129921).

## Setup

Create the venv and install dependencies — see [`Cohere_Arabic_Training_Setup.txt`](Cohere_Arabic_Training_Setup.txt) for the full walkthrough, or just:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Training script

[`train_cohere_full.py`](train_cohere_full.py) does the actual fine-tuning (full FT, fp32 weights + bf16 autocast, gradient checkpointing). It's launched via `torchrun` across multiple GPUs — see [`example.sh`](example.sh) for a template launch script.

Run it (edit `TRAIN_CSV` / `MODEL_PATH` in the script first):

```bash
chmod +x example.sh
setsid nohup bash example.sh \
  > example.log 2>&1 < /dev/null &
echo "PID: $!"
```

`example.sh` uses 4 GPUs (`torchrun --nproc_per_node=4`), and expects a pipe-delimited CSV (`audio_path|transcript`).

## Data

All training data lives in GCS:

- **`gs://speech-annotation-source/stt-data/STT-Training-Data-sep2026`**
 — dataset #1, best checkpoints came from this.
  Example CSV: `gs://speech-annotation-source/stt-data/STT-Training-Data-sep2026/org/stt-data/Dana/Data/CSVs_4_training/15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches.csv`

  > Note: Cohere transcribes digits **as numerals** (e.g. `123`), not spelled out in words (e.g. "one two three"). When preparing/normalizing transcripts for this dataset, keep digits as-is rather than converting them to words.

  
- **`gs://speech-annotation-source/stt-data/Dana/`** and **`gs://speech-annotation-source/stt-data/Data/`**

 — dataset #2, all data.
  Best subset (2nd-best checkpoints): `gs://speech-annotation-source/stt-data/Dana/Data/CSVs_4_training/digitsinWords/training_lahgtna+crtvai+batches+cv18+masc+voxpop+dialects+tts_284k_14Aug26.csv`
