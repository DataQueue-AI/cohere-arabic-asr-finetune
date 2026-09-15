"""FULL fine-tune of CohereLabs/cohere-transcribe-arabic-07-2026 (2.07B,
cohere_asr, Conformer encoder-decoder) on dialectal Arabic — meant to run on a
LARGE GPU (≈48–80 GB) or multi-GPU, unlike the 32 GB LoRA variant.

Memory: the model loads in **fp32** + **bf16 autocast** (stable full-FT), with
gradient checkpointing. Rough single-GPU budget for 2B params:
  ~8 GB weights + 8 GB grads + ~16 GB AdamW state + activations ≈ 45–55 GB.
For smaller GPUs use DeepSpeed ZeRO-2/3 (`--deepspeed ds_zero2.json`) or 8-bit Adam.

Setup (see setup.sh / requirements.txt):
  pip install -r requirements.txt         # transformers with cohere_asr + torch (CUDA)
  huggingface-cli login                   # needs access to the GATED Cohere model
                                          # + the (private) dataset

Run (single GPU):
  python train_cohere_full.py --output_dir cohere-ar-full --push_to_hub oddadmix/cohere-arabic-full-ft

Run (multi-GPU / DeepSpeed):
  accelerate launch train_cohere_full.py --deepspeed ds_zero2.json ...

⚠️ This model is a strong Arabic specialist; in our 32 GB LoRA run it *overfit*
this data (base 0.457 → 0.510 WER). Full FT can overfit faster — use a LOW LR,
watch **eval WER** (not loss), and keep the best checkpoint. Defaults are conservative.
"""
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import jiwer
from datasets import Audio, load_dataset
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from normalize import clean_text

DATASET = "oddadmix/dialectal-arabic-lahgtna-v2-smaller-augmented"
MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
SR = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--dataset", default=None,
                   help="HF Hub dataset id. Ignored when --csv_file is set.")
    p.add_argument("--csv_file", default=None,
                   help="Pipe-delimited CSV: wav_path|transcription (local).")
    p.add_argument("--eval_csv_file", default=None,
                   help="Optional eval CSV in the same format.")
    p.add_argument("--language", default="ar")
    p.add_argument("--output_dir", default="cohere-ar-full")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)   # low: strong base
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)           # -1 = use epochs
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_audio_seconds", type=float, default=30.0)
    p.add_argument("--max_text_chars", type=int, default=1000)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--deepspeed", default=None, help="path to a DeepSpeed config json")
    p.add_argument("--push_to_hub", default=None,
                   help="HF repo id to push the final model to (private).")
    p.add_argument("--final_eval_clips", type=int, default=932,
                   help="how many test clips to score WER/CER on at the end (0 = skip).")
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--save_only_model", action="store_true",
                   help="Skip optimizer/scheduler state in checkpoints (saves ~16 GB per ckpt).")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="resume training: 'auto' (latest checkpoint in output_dir) or a "
                        "specific checkpoint dir. Omit to start fresh.")
    return p.parse_args()


def decode_audio(a: Any) -> np.ndarray:
    if isinstance(a, dict) and a.get("array") is not None:
        arr = np.asarray(a["array"], dtype=np.float32); sr = a.get("sampling_rate", SR)
    else:
        raw = a["bytes"] if a.get("bytes") is not None else a["path"]
        src = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else raw
        arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(arr, dtype=np.float32)


@dataclass
class CohereCollator:
    processor: Any
    language: str

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        arrays = [decode_audio(ex["audio"]) for ex in batch]
        texts = [clean_text(ex["text"]) for ex in batch]
        enc = self.processor(arrays, text=texts, sampling_rate=SR,
                             language=self.language, return_tensors="pt", padding=True)
        pad = self.processor.tokenizer.pad_token_id
        prompt, trans = enc["decoder_input_ids"], enc["labels"]
        # causal-LM loss needs aligned decoder_input_ids/labels: concatenate the
        # language prompt + transcript; supervise only the transcript.
        dec_in = torch.cat([prompt, trans], dim=1)
        labels = torch.cat([torch.full_like(prompt, -100), trans], dim=1).masked_fill(
            dec_in == pad, -100)
        out = {"input_features": enc["input_features"], "decoder_input_ids": dec_in,
               "decoder_attention_mask": (dec_in != pad).long(), "labels": labels}
        if "attention_mask" in enc:
            out["attention_mask"] = enc["attention_mask"]
        return out


def load_split(name, args):
    ds = load_dataset(args.dataset, split=name).cast_column("audio", Audio(decode=False))

    def keep(text, duration):
        t = clean_text(text)
        return bool(t) and len(t) <= args.max_text_chars and (
            duration is None or 0.5 <= duration <= args.max_audio_seconds)

    return ds.filter(keep, input_columns=["text", "duration"], num_proc=args.num_workers)


def load_csv_split(csv_path: str, args) -> Any:
    """Load a pipe-delimited CSV (wav_path|transcription) for CohereCollator."""
    from datasets import Dataset as _Dataset
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            wav_path, text = parts[0].strip(), parts[1].strip()
            t = clean_text(text)
            if not t or len(t) > args.max_text_chars:
                continue
            if not Path(wav_path).is_file():
                continue
            try:
                info = sf.info(wav_path)
                dur = info.frames / info.samplerate
            except Exception:
                continue
            if dur < 0.5 or dur > args.max_audio_seconds:
                continue
            rows.append({"audio": {"path": wav_path}, "text": text})
    print(f"CSV loaded {len(rows)} samples from {csv_path}")
    return _Dataset.from_list(rows)


@torch.no_grad()
def eval_wer(model, processor, ds, language, n):
    """Per-sample generation WER/CER (batched generation garbles this model)."""
    model.eval()
    preds, refs = [], []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        inp = processor(decode_audio(ex["audio"]), sampling_rate=SR, language=language,
                        return_tensors="pt").to(model.device)
        inp["input_features"] = inp["input_features"].to(model.dtype)
        g = model.generate(**inp, max_new_tokens=256)
        preds.append(clean_text(processor.tokenizer.batch_decode(g, skip_special_tokens=True)[0]))
        refs.append(clean_text(ex["text"]))
    pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    preds, refs = map(list, zip(*pairs))
    return jiwer.wer(refs, preds), jiwer.cer(refs, preds)


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    # fp32 load + bf16 autocast = stable full fine-tune (params stay fp32).
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32)
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.csv_file:
        train_ds = load_csv_split(args.csv_file, args)
        eval_ds = load_csv_split(args.eval_csv_file, args) if args.eval_csv_file else None
    else:
        train_ds, eval_ds = load_split("train", args), load_split("test", args)
    print({"train": len(train_ds), "test": len(eval_ds) if eval_ds is not None else 0})
    collator = CohereCollator(processor=processor, language=args.language)

    has_eval = eval_ds is not None
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=True, fp16=False,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.eval_steps,
        logging_steps=25, report_to=[], predict_with_generate=False,
        save_total_limit=args.save_total_limit, save_only_model=args.save_only_model,
        load_best_model_at_end=has_eval,
        metric_for_best_model="loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        dataloader_num_workers=args.num_workers, remove_unused_columns=False,
        label_names=["labels"], deepspeed=args.deepspeed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, processing_class=processor,
    )
    # resume support: 'auto' lets the Trainer pick the latest checkpoint in output_dir.
    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # final WER/CER (per-sample generation, correct for this model)
    wer = cer = None
    if args.final_eval_clips and eval_ds is not None:
        try:
            wer, cer = eval_wer(model, processor, eval_ds, args.language, args.final_eval_clips)
            print(f"FINAL clean_text WER {wer:.4f}  CER {cer:.4f}  on {min(args.final_eval_clips, len(eval_ds))} clips")
        except Exception as e:
            print(f"(final eval skipped: {e})")

    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    best = min(evals, key=lambda h: h["eval_loss"]) if evals else {}
    summary = {
        "run_name": Path(args.output_dir).name, "base_model": args.model_id,
        "dataset": args.csv_file or args.dataset, "method": "FULL fine-tune (fp32 + bf16 autocast)",
        "learning_rate": args.learning_rate, "epochs": args.num_train_epochs,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "best_eval_loss": round(best.get("eval_loss", float("nan")), 4) if evals else None,
        "final_wer": round(wer, 4) if wer is not None else None,
        "final_cer": round(cer, 4) if cer is not None else None,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("summary:", summary)

    if args.push_to_hub:
        print(f"pushing to {args.push_to_hub} (private) ...")
        model.push_to_hub(args.push_to_hub, private=True)
        processor.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed -> https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
