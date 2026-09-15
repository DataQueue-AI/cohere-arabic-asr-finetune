"""Inference script for Cohere ASR finetuned model."""
import os
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from safetensors.torch import load_file


def _build_prefix_maps(path_map_from=None, path_map_to=None):
    if not path_map_from and not path_map_to:
        return []
    if not path_map_from or not path_map_to:
        raise ValueError("Both --path_map_from and --path_map_to must be provided together")
    if len(path_map_from) != len(path_map_to):
        raise ValueError(
            f"--path_map_from count ({len(path_map_from)}) must match --path_map_to count ({len(path_map_to)})"
        )
    pairs = list(zip(path_map_from, path_map_to))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def resolve_data_path(path, prefix_maps=None, rewrite_gs_to_mount=False, gcs_mount_root="/gcs"):
    resolved = path.strip()

    if rewrite_gs_to_mount and resolved.startswith("gs://"):
        bucket_and_object = resolved[len("gs://"):].lstrip("/")
        resolved = os.path.join(gcs_mount_root.rstrip("/"), bucket_and_object)

    if prefix_maps:
        for src_prefix, dst_prefix in prefix_maps:
            if resolved.startswith(src_prefix):
                resolved = dst_prefix + resolved[len(src_prefix):]
                break

    return os.path.expandvars(os.path.expanduser(resolved))


def load_model(base_model_path, weights_path=None, device="cuda:3"):
    """Load model with correct weight loading (workaround for transformers 5.12 bug).
    
    Args:
        base_model_path: Path to the base model (e.g. './cohere-transcribe-03-2026')
        weights_path: Path to finetuned model.safetensors (None = use base model)
        device: CUDA device
    """
    import transformers as _tf
    _tv = tuple(int(x) for x in _tf.__version__.split(".")[:2])

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Manually reload weights only on transformers 5.12 which corrupts ~60% of values.
    # On other versions from_pretrained handles key remapping correctly.
    if _tv == (5, 12):
        ckpt_path = weights_path or f"{base_model_path}/model.safetensors"
        state = load_file(ckpt_path)
        state = {k: v for k, v in state.items() if not k.startswith("preprocessor.")}
        model.load_state_dict(state, strict=True)

    model = model.to(device).eval()
    return model, processor


def generate(model, processor, audio_path, language="ar", device="cuda:3"):
    """Transcribe a single audio file.
    
    Args:
        model: Loaded model
        processor: Loaded processor
        audio_path: Path to audio file (any sample rate, will be resampled to 16kHz)
        language: Language code (ar, en, fr, etc.)
        device: CUDA device
    """
    audio, sr = librosa.load(audio_path, sr=16000)

    # Some remote-code model classes expose `transcribe`, others only support
    # processor + generate. Keep this path compatible with both.
    if hasattr(model, "transcribe"):
        result = model.transcribe(
            processor=processor,
            language=language,
            audio_arrays=[audio],
            sample_rates=[16000],
        )
        return result[0]

    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        language=language,
        return_tensors="pt",
        punctuation=True,
    )

    model_inputs = {}
    if "input_features" in inputs:
        model_inputs["input_features"] = inputs["input_features"].to(device)
    if "decoder_input_ids" in inputs:
        model_inputs["decoder_input_ids"] = inputs["decoder_input_ids"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_length=225)

    # If prompt tokens are present in generated output, strip them.
    if "decoder_input_ids" in model_inputs:
        prompt_len = model_inputs["decoder_input_ids"].shape[1]
        generated_ids = generated_ids[:, prompt_len:]

    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--base_model", default="./cohere-transcribe-03-2026")
    parser.add_argument("--weights", default="./cohere-ft/model.safetensors",
                        help="Path to finetuned weights (default: ./cohere-ft/model.safetensors)")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument(
        "--path_map_from",
        nargs="+",
        default=None,
        help="One or more source prefixes to rewrite in paths",
    )
    parser.add_argument(
        "--path_map_to",
        nargs="+",
        default=None,
        help="One or more destination prefixes (same count/order as --path_map_from)",
    )
    parser.add_argument(
        "--rewrite_gs_to_mount",
        action="store_true",
        default=False,
        help="Rewrite gs://bucket/path to local mount path: <gcs_mount_root>/bucket/path",
    )
    parser.add_argument(
        "--gcs_mount_root",
        default="/gcs",
        help="Mount root used with --rewrite_gs_to_mount (default: /gcs)",
    )
    args = parser.parse_args()

    prefix_maps = _build_prefix_maps(args.path_map_from, args.path_map_to)
    base_model_path = resolve_data_path(
        args.base_model,
        prefix_maps=prefix_maps,
        rewrite_gs_to_mount=args.rewrite_gs_to_mount,
        gcs_mount_root=args.gcs_mount_root,
    )
    weights_path = resolve_data_path(
        args.weights,
        prefix_maps=prefix_maps,
        rewrite_gs_to_mount=args.rewrite_gs_to_mount,
        gcs_mount_root=args.gcs_mount_root,
    )
    audio_path = resolve_data_path(
        args.audio,
        prefix_maps=prefix_maps,
        rewrite_gs_to_mount=args.rewrite_gs_to_mount,
        gcs_mount_root=args.gcs_mount_root,
    )

    model, processor = load_model(base_model_path, weights_path, args.device)
    text = generate(model, processor, audio_path, args.language, args.device)
    print(text)
