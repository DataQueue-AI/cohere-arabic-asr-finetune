"""Precompute audio durations for a pipe-delimited training CSV, writing a 3rd
column so train_cohere_full.py can filter by min/max length WITHOUT any network
call at training-startup time.

For gs:// rows this reads only the first ~1 MiB of each object (a Range GET),
which is enough to parse a standard WAV header -- no full-file download needed.
Falls back to a full download only if the header parse fails.

Usage:
  python precompute_gcs_durations.py in.csv out.csv \
      --key_file /workspace/google.json --workers 64

Output format: wav_path|transcription|duration_seconds
(local wav_path rows are also re-emitted with their duration, unchanged otherwise)
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf

HEADER_BYTES = 1 << 20  # 1 MiB: plenty for a WAV header even with extra LIST/INFO chunks
_local = threading.local()


def _client(key_file: str | None):
    if not hasattr(_local, "client"):
        from google.cloud import storage
        _local.client = (storage.Client.from_service_account_json(key_file)
                          if key_file else storage.Client())
    return _local.client


def gcs_duration(gs_uri: str, key_file: str | None) -> float | None:
    bucket, _, key = gs_uri[len("gs://"):].partition("/")
    blob = _client(key_file).bucket(bucket).blob(key)
    try:
        head = blob.download_as_bytes(start=0, end=HEADER_BYTES - 1)
        info = sf.info(io.BytesIO(head))
        return info.frames / info.samplerate
    except Exception:
        try:
            data = blob.download_as_bytes()  # rare fallback: full download
            info = sf.info(io.BytesIO(data))
            return info.frames / info.samplerate
        except Exception:
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--key_file", default="/workspace/google.json")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    rows = []
    with open(args.in_csv, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            rows.append((parts[0].strip(), parts[1].strip()))
    print(f"{len(rows)} rows to process", file=sys.stderr)

    results: list[tuple[str, str, float | None] | None] = [None] * len(rows)

    def work(i: int, path: str, text: str):
        if path.startswith("gs://"):
            dur = gcs_duration(path, args.key_file)
        else:
            try:
                info = sf.info(path)
                dur = info.frames / info.samplerate
            except Exception:
                dur = None
        return i, path, text, dur

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, p, t) for i, (p, t) in enumerate(rows)]
        done = 0
        for fut in as_completed(futs):
            i, path, text, dur = fut.result()
            results[i] = (path, text, dur)
            done += 1
            if done % 5000 == 0:
                print(f"{done}/{len(rows)}", file=sys.stderr)

    n_ok = 0
    with open(args.out_csv, "w", encoding="utf-8") as out:
        for entry in results:
            path, text, dur = entry
            if dur is None:
                continue
            out.write(f"{path}|{text}|{dur:.3f}\n")
            n_ok += 1
    print(f"wrote {n_ok}/{len(rows)} rows with duration -> {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
