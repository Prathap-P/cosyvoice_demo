#!/usr/bin/env python3
"""
eval_audio.py — CosyVoice 2-0.5B Benchmarking Script
======================================================
Uses the `cosyvoice2-eu` pip package (no source clone required).
Benchmarks FunAudioLLM/CosyVoice2-0.5B across 5 style variants + 1 clone.

Install:
    pip install cosyvoice2-eu torchaudio

Metrics:
    RTF (Real-Time Factor) = Inference Time / Audio Duration
    RTF < 1.0 → faster than real-time (good)
    RTF > 1.0 → slower than real-time (slow)

Usage:
    python eval_audio.py
    python eval_audio.py --ref-wav reference.wav --ref-text "transcript here"
    python eval_audio.py --model-repo FunAudioLLM/CosyVoice2-0.5B
"""

import sys
import time
import argparse
import platform
import datetime
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. DEPENDENCY CHECKS
# ---------------------------------------------------------------------------
def check_dependencies() -> dict:
    """Verify runtime dependencies. Exits with a clear message on failure."""
    status = {}

    try:
        import torch
        status["torch"] = torch.__version__
        cuda_ok = torch.cuda.is_available()
        mps_ok  = (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if cuda_ok:
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
            status["gpu"] = f"CUDA — {name} ({vram:.1f} GB)"
        elif mps_ok:
            status["gpu"] = "MPS (Apple Silicon)"
        else:
            status["gpu"] = "CPU only"
        status["cuda"] = cuda_ok
        status["mps"]  = mps_ok
    except ImportError:
        print("ERROR: PyTorch not found.")
        print("  pip install torch torchaudio")
        sys.exit(1)

    try:
        import torchaudio
        status["torchaudio"] = torchaudio.__version__
    except ImportError:
        print("ERROR: torchaudio not found.  pip install torchaudio")
        sys.exit(1)

    try:
        from cosyvoice2_eu import load  # noqa: F401
        status["cosyvoice2_eu"] = "OK"
    except ImportError:
        print("ERROR: cosyvoice2-eu not found.")
        print("  pip install cosyvoice2-eu")
        sys.exit(1)

    return status


# ---------------------------------------------------------------------------
# 2. RTF
# ---------------------------------------------------------------------------
def compute_rtf(audio_tensor, sample_rate: int, inference_time_s: float) -> float:
    """RTF = Inference Time / Audio Duration.  < 1.0 is faster than real-time."""
    n_samples = audio_tensor.shape[-1]
    audio_dur = n_samples / sample_rate
    if audio_dur == 0:
        return float("inf")
    return inference_time_s / audio_dur


# ---------------------------------------------------------------------------
# 3. RESULTS LOGGER
# ---------------------------------------------------------------------------
class ResultsLogger:
    def __init__(self, path: Path, model_repo: str, device: str):
        self.path       = path
        self.entries    = []
        self.model_repo = model_repo
        self.device     = device
        self.ts         = datetime.datetime.now().isoformat(timespec="seconds")

    def log(self, name, mode, audio_s, infer_s, rtf, fname, notes=""):
        self.entries.append(dict(
            name=name, mode=mode,
            audio_s=round(audio_s, 3), infer_s=round(infer_s, 3),
            rtf=round(rtf, 4), fname=fname, notes=notes,
        ))

    def log_error(self, name, mode, err):
        self.entries.append(dict(
            name=name, mode=mode,
            audio_s="—", infer_s="—", rtf="—",
            fname="FAILED", notes=f"ERROR: {err}",
        ))

    def save(self):
        lines = [
            "# CosyVoice 2-0.5B Benchmark Results", "",
            f"**Generated:** {self.ts}  ",
            f"**Model:** `{self.model_repo}`  ",
            f"**Device:** {self.device}  ",
            f"**Python:** {sys.version.split()[0]}  ",
            f"**Platform:** {platform.platform()}  ", "",
            "> **RTF** = Inference Time / Audio Duration  ",
            "> RTF < 1.0 → faster than real-time ✅  |  RTF > 1.0 → slower ⚠️",
            "", "---", "", "## Results", "",
            "| # | Name | Mode | Audio (s) | Infer (s) | RTF | File | Notes |",
            "|---|------|------|----------:|----------:|-----|------|-------|",
        ]
        for i, e in enumerate(self.entries, 1):
            if isinstance(e["rtf"], float):
                icon    = "✅" if e["rtf"] < 1.0 else "⚠️"
                rtf_str = f"`{e['rtf']:.4f}` {icon}"
            else:
                rtf_str = str(e["rtf"])
            lines.append(
                f"| {i} | {e['name']} | {e['mode']} "
                f"| {e['audio_s']} | {e['infer_s']} "
                f"| {rtf_str} | `{e['fname']}` | {e['notes']} |"
            )

        rtfs = [e["rtf"] for e in self.entries if isinstance(e["rtf"], float)]
        if rtfs:
            avg = sum(rtfs) / len(rtfs)
            lines += [
                "", "---", "", "## Summary", "",
                "| Metric | Value |", "|--------|-------|",
                f"| Completed | {len(rtfs)} / {len(self.entries)} |",
                f"| Avg RTF   | `{avg:.4f}` |",
                f"| Best RTF  | `{min(rtfs):.4f}` |",
                f"| Worst RTF | `{max(rtfs):.4f}` |",
            ]

        lines += ["", "---", "", "*Generated by eval_audio.py*"]
        self.path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[results] Saved → {self.path}")


# ---------------------------------------------------------------------------
# 4. BENCHMARK RUNS DEFINITION
# ---------------------------------------------------------------------------
# cosyvoice2-eu exposes only two methods:
#   tts(text, prompt, *, speed, text_frontend) → (Tensor, int)
#   stream(text, prompt, *, speed, text_frontend) → Iterator[Tensor]
#
# Both use cross_lingual inference internally (reference audio, no transcript).
# We differentiate 5 "voices" via speed + text_frontend variations and a
# warm vs cold run, plus the user-supplied reference.wav for run 6.

VOICE_CONFIGS = [
    {
        "id":           "v1_standard",
        "label":        "Standard (speed=1.0)",
        "speed":        1.0,
        "text_frontend": False,
    },
    {
        "id":           "v2_slow",
        "label":        "Slow (speed=0.8)",
        "speed":        0.8,
        "text_frontend": False,
    },
    {
        "id":           "v3_fast",
        "label":        "Fast (speed=1.3)",
        "speed":        1.3,
        "text_frontend": False,
    },
    {
        "id":           "v4_frontend_on",
        "label":        "Standard + text_frontend=True",
        "speed":        1.0,
        "text_frontend": True,
    },
    {
        "id":           "v5_warm",
        "label":        "Warm run (speed=1.0, repeat)",
        "speed":        1.0,
        "text_frontend": False,
    },
]


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def run_benchmark(args):
    import torchaudio
    from cosyvoice2_eu import load

    print("\n" + "=" * 60)
    print("  CosyVoice 2-0.5B — Benchmarking Suite (cosyvoice2-eu)")
    print("=" * 60)

    # ── Dependency check ────────────────────────────────────────────────────
    dep = check_dependencies()
    print(f"[env] PyTorch      : {dep['torch']}")
    print(f"[env] torchaudio   : {dep['torchaudio']}")
    print(f"[env] cosyvoice2-eu: {dep['cosyvoice2_eu']}")
    print(f"[env] Device       : {dep['gpu']}")

    device_label = dep["gpu"]

    # ── Read input text ─────────────────────────────────────────────────────
    text_path = Path(args.input).resolve()
    if not text_path.exists():
        print(f"ERROR: input file not found: {text_path}")
        sys.exit(1)
    tts_text = text_path.read_text(encoding="utf-8").strip()
    if not tts_text:
        print("ERROR: input_text.txt is empty")
        sys.exit(1)
    print(f"[input] {len(tts_text)} chars: {tts_text[:80]}{'...' if len(tts_text) > 80 else ''}")

    # ── Reference WAV (required — cosyvoice2-eu is zero-shot only) ──────────
    ref_wav = Path(args.ref_wav).resolve()
    if not ref_wav.exists():
        print(f"\nERROR: Reference WAV not found: {ref_wav}")
        print()
        print("cosyvoice2-eu requires a reference WAV for ALL synthesis.")
        print("Provide a 3–30s, mono, 16kHz WAV file:")
        print(f"  --ref-wav /path/to/speaker.wav")
        sys.exit(1)
    print(f"[input] Reference  : {ref_wav}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n[model] Loading via cosyvoice2-eu → {args.model_repo}")
    print("[model] First run downloads weights (~2 GB) to ~/.cache/cosyvoice2-eu …")
    t_load = time.perf_counter()
    try:
        cosy = load(repo_id=args.model_repo)
    except Exception as exc:
        print(f"\nERROR: Model load failed: {exc}")
        traceback.print_exc()
        sys.exit(1)
    print(f"[model] Ready in {time.perf_counter() - t_load:.1f}s  |  sample_rate={cosy.sample_rate} Hz")

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = ResultsLogger(SCRIPT_DIR / "results.md", args.model_repo, device_label)

    total_runs = len(VOICE_CONFIGS) + 1
    print(f"\n{'─'*60}")
    print(f" {total_runs} runs  |  ref: {ref_wav.name}")
    print(f"{'─'*60}")

    # ── Runs 1–5: style variants ────────────────────────────────────────────
    for idx, cfg in enumerate(VOICE_CONFIGS, 1):
        print(f"\n[{idx}/{total_runs}] {cfg['label']}")

        out_file = out_dir / f"output_{cfg['id']}.wav"
        try:
            t0          = time.perf_counter()
            wav, sr     = cosy.tts(
                tts_text,
                str(ref_wav),
                speed=cfg["speed"],
                text_frontend=cfg["text_frontend"],
            )
            infer_time  = time.perf_counter() - t0

            torchaudio.save(str(out_file), wav, sr)

            audio_dur   = wav.shape[-1] / sr
            rtf         = compute_rtf(wav, sr, infer_time)
            icon        = "✅" if rtf < 1.0 else "⚠️"
            print(f"  ✓  audio={audio_dur:.2f}s  infer={infer_time:.2f}s  RTF={rtf:.4f} {icon}")
            print(f"  →  {out_file.name}")

            logger.log(cfg["id"], "cross_lingual", audio_dur, infer_time, rtf,
                       out_file.name, notes=f"speed={cfg['speed']}")

        except Exception as exc:
            print(f"  ✗  FAILED: {exc}")
            traceback.print_exc()
            logger.log_error(cfg["id"], "cross_lingual", str(exc))

    # ── Run 6: user clone with separate reference ───────────────────────────
    clone_ref = Path(args.clone_ref).resolve() if args.clone_ref else ref_wav
    print(f"\n[{total_runs}/{total_runs}] Voice Clone — {clone_ref.name}")

    out_file = out_dir / "output_voice_clone.wav"
    try:
        t0         = time.perf_counter()
        wav, sr    = cosy.tts(tts_text, str(clone_ref), speed=1.0, text_frontend=False)
        infer_time = time.perf_counter() - t0

        torchaudio.save(str(out_file), wav, sr)

        audio_dur  = wav.shape[-1] / sr
        rtf        = compute_rtf(wav, sr, infer_time)
        icon       = "✅" if rtf < 1.0 else "⚠️"
        print(f"  ✓  audio={audio_dur:.2f}s  infer={infer_time:.2f}s  RTF={rtf:.4f} {icon}")
        print(f"  →  {out_file.name}")

        logger.log("voice_clone", "cross_lingual_clone",
                   audio_dur, infer_time, rtf, out_file.name,
                   notes=f"ref={clone_ref.name}")

    except Exception as exc:
        print(f"  ✗  FAILED: {exc}")
        traceback.print_exc()
        logger.log_error("voice_clone", "cross_lingual_clone", str(exc))

    # ── Save & summarise ────────────────────────────────────────────────────
    logger.save()

    rtfs = [e["rtf"] for e in logger.entries if isinstance(e["rtf"], float)]
    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    if rtfs:
        avg = sum(rtfs) / len(rtfs)
        print(f"  Completed : {len(rtfs)} / {len(logger.entries)}")
        print(f"  Avg RTF   : {avg:.4f}  {'✅' if avg < 1.0 else '⚠️'}")
        print(f"  Best RTF  : {min(rtfs):.4f}   Worst RTF: {max(rtfs):.4f}")
    print(f"  Results   : {SCRIPT_DIR / 'results.md'}")
    print(f"  Outputs   : {out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CosyVoice 2-0.5B benchmark (cosyvoice2-eu)")
    p.add_argument(
        "--model-repo",
        default="FunAudioLLM/CosyVoice2-0.5B",
        help="HuggingFace repo ID. Default: FunAudioLLM/CosyVoice2-0.5B",
    )
    p.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "input_text.txt"),
        help="Path to input text file.",
    )
    p.add_argument(
        "--ref-wav",
        required=True,
        help="Path to reference WAV (3–30s, mono, ≥16kHz). REQUIRED.",
    )
    p.add_argument(
        "--clone-ref",
        default=None,
        help="Optional second reference WAV for the clone run. "
             "Defaults to --ref-wav if not provided.",
    )
    p.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Directory for output WAV files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
