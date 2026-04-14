#!/usr/bin/env python3
"""
================================================================================
  CosyVoice 2-0.5B  —  Quick-Look Evaluation Demo
  Model: FunAudioLLM/CosyVoice2-0.5B (ModelScope: iic/CosyVoice2-0.5B)
  Supports: Zero-shot cloning · Cross-lingual · Instruct mode · Fine-grained control
================================================================================

ENVIRONMENT SETUP (run once before this script):
─────────────────────────────────────────────────
  # 1. Create & activate venv (already done if you ran Wibey's setup)
  python3 -m venv .venv --system-site-packages
  source .venv/bin/activate

  # 2. Core packages (via Walmart Artifactory mirror)
  PIP="pip install -i https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/devtools-pypi/simple/ --trusted-host pypi.ci.artifacts.walmart.com"
  $PIP torch torchaudio transformers huggingface_hub modelscope
  $PIP soundfile numpy scipy librosa omegaconf einops diffusers
  $PIP inflect unidecode onnxruntime "ruamel.yaml>=0.17.28,<0.19.0"

  # 3. HyperPyYAML (from source — Artifactory doesn't proxy its wheel download)
  git clone --depth=1 https://github.com/speechbrain/HyperPyYAML.git /tmp/hyperpyyaml_src
  pip install --no-index /tmp/hyperpyyaml_src

  # 4. CosyVoice library (must be in same directory as this script)
  git clone --depth=1 https://github.com/FunAudioLLM/CosyVoice.git cosyvoice_repo

  # 5. Download model weights (choose one):
  #    Option A — ModelScope (recommended for first run, no HF token needed):
  python3 -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
  #    Option B — HuggingFace Hub:
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

INPUTS:
  • input_text.txt   — text to synthesise  (auto-created if missing)
  • reference.wav    — optional reference audio for zero-shot voice cloning
    (16 kHz mono, 3–30 s of clean speech; must include a matching transcript
     in REFERENCE_TRANSCRIPT below)

OUTPUTS:
  output_zero_shot_cloned.wav      — zero-shot cloned voice (needs reference.wav)
  output_cross_lingual.wav         — cross-lingual synthesis using reference voice
  output_instruct_natural.wav      — instruct mode: natural conversational style
  output_instruct_excited.wav      — instruct mode: excited/energetic delivery
  output_instruct_slow_clear.wav   — instruct mode: slow and clear narration
  output_fine_grained.wav          — fine-grained control (laughter, breath marks)
================================================================================
"""

import os
import sys
import time
import wave
import logging
import warnings
import contextlib
from pathlib import Path
from typing import Optional

# ── Suppress noisy third-party warnings ──────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent.resolve()
COSYVOICE_REPO   = SCRIPT_DIR / "cosyvoice_repo"
MATCHA_TTS_DIR   = COSYVOICE_REPO / "third_party" / "Matcha-TTS"
MODEL_DIR        = SCRIPT_DIR / "pretrained_models" / "CosyVoice2-0.5B"
INPUT_TEXT_FILE  = SCRIPT_DIR / "input_text.txt"
REFERENCE_WAV    = SCRIPT_DIR / "reference.wav"
OUTPUT_DIR       = SCRIPT_DIR / "outputs"

# ── Reference audio transcript ─────────────────────────────────────────────
# IMPORTANT: Edit this to match what is spoken in your reference.wav file.
# If no reference.wav is present, zero-shot cloning tests will be skipped.
REFERENCE_TRANSCRIPT = (
    "I hope you can do better than me in the future."
)

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s │ %(message)s"
)
log = logging.getLogger("cosyvoice_demo")
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(title: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def get_wav_duration_seconds(path: str) -> float:
    """Return duration of a WAV file in seconds."""
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def save_audio_chunks(chunks_iter, output_path: str, sample_rate: int) -> Optional[float]:
    """
    Collect all audio chunks from a CosyVoice generator, concatenate, and save.
    Returns the total audio duration in seconds, or None on failure.
    """
    import torch
    import torchaudio

    all_chunks = []
    for chunk in chunks_iter:
        all_chunks.append(chunk["tts_speech"])

    if not all_chunks:
        log.warning(f"No audio chunks generated for {output_path}")
        return None

    audio = torch.cat(all_chunks, dim=1)  # [1, T]
    torchaudio.save(output_path, audio, sample_rate)
    duration = audio.shape[1] / sample_rate
    return duration


def print_metrics(label: str, inference_time: float, audio_duration: float) -> None:
    """Print TTFA and RTF metrics for a synthesis run."""
    rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")
    print(f"  ⏱  TTFA / total inference : {inference_time:.2f}s")
    print(f"  🎵 Audio duration         : {audio_duration:.2f}s")
    print(f"  ⚡ Real-Time Factor (RTF)  : {rtf:.3f}  "
          f"({'faster' if rtf < 1.0 else 'slower'} than real-time)")


def run_synthesis(label: str, gen_fn, output_path: str, sample_rate: int) -> bool:
    """
    Run a synthesis function, time it, save output, and print metrics.
    Returns True on success.
    """
    print(f"\n  ▶  {label}")
    try:
        t0 = time.perf_counter()
        duration = save_audio_chunks(gen_fn(), output_path, sample_rate)
        inference_time = time.perf_counter() - t0

        if duration is None:
            print(f"  ✗  Failed — no audio generated")
            return False

        print(f"  ✓  Saved → {Path(output_path).name}")
        print_metrics(label, inference_time, duration)
        return True

    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        log.debug("Synthesis error", exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Environment bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def setup_paths() -> bool:
    """Add CosyVoice repo and Matcha-TTS to sys.path."""
    if not COSYVOICE_REPO.exists():
        log.error(
            f"CosyVoice repo not found at {COSYVOICE_REPO}\n"
            "  Run: git clone --depth=1 https://github.com/FunAudioLLM/CosyVoice.git cosyvoice_repo"
        )
        return False

    for p in [str(COSYVOICE_REPO), str(MATCHA_TTS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return True


def load_model():
    """
    Load CosyVoice2-0.5B. Downloads weights automatically via ModelScope
    if the local model directory does not exist.
    """
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_dir = str(MODEL_DIR)

    if not MODEL_DIR.exists():
        log.info("Model weights not found locally — downloading via ModelScope …")
        log.info("(This may take a few minutes on first run)")
        try:
            from modelscope import snapshot_download
            downloaded = snapshot_download(
                "iic/CosyVoice2-0.5B",
                local_dir=model_dir
            )
            log.info(f"Downloaded to: {downloaded}")
        except Exception as e:
            log.error(
                f"ModelScope download failed: {e}\n"
                "  Try manually:\n"
                "  python3 -c \"from modelscope import snapshot_download; "
                "snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\""
            )
            return None

    log.info(f"Loading CosyVoice2-0.5B from {model_dir} …")
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    log.info("Model loaded successfully ✓")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Synthesis demos
# ─────────────────────────────────────────────────────────────────────────────

def demo_zero_shot_cloning(model, tts_text: str, output_dir: Path) -> None:
    """
    Zero-shot voice cloning: synthesise tts_text using the acoustic identity
    extracted from reference.wav, without any fine-tuning.
    """
    print_banner("ZERO-SHOT VOICE CLONING  (reference.wav → cloned voice)")

    if not REFERENCE_WAV.exists():
        print("  ⚠  reference.wav not found — skipping cloning demos.")
        print("     Place a 3-30s 16kHz mono WAV at:", REFERENCE_WAV)
        return

    ref_path = str(REFERENCE_WAV)
    ref_text = REFERENCE_TRANSCRIPT

    # ── 1. Standard zero-shot cloning ────────────────────────────────────────
    run_synthesis(
        "Zero-shot clone  [standard]",
        lambda: model.inference_zero_shot(
            tts_text, ref_text, ref_path, stream=False
        ),
        str(output_dir / "output_zero_shot_cloned.wav"),
        model.sample_rate,
    )

    # ── 2. Cross-lingual: use reference voice but speak the tts_text language ─
    run_synthesis(
        "Cross-lingual  [reference voice + English text]",
        lambda: model.inference_cross_lingual(
            tts_text, ref_path, stream=False
        ),
        str(output_dir / "output_cross_lingual.wav"),
        model.sample_rate,
    )


def demo_instruct_modes(model, tts_text: str, output_dir: Path) -> None:
    """
    Instruct mode (CosyVoice2-specific): synthesise tts_text with different
    style/emotion instructions passed as natural language prompts.
    Requires reference.wav for the speaker identity.
    """
    print_banner("INSTRUCT MODE  (style-guided synthesis)")

    if not REFERENCE_WAV.exists():
        print("  ⚠  reference.wav not found — skipping instruct demos.")
        return

    ref_path = str(REFERENCE_WAV)

    instruct_configs = [
        (
            "Natural / conversational",
            "Please speak in a natural, conversational tone.<|endofprompt|>",
            "output_instruct_natural.wav",
        ),
        (
            "Excited / energetic",
            "Please deliver this with enthusiasm and energy!<|endofprompt|>",
            "output_instruct_excited.wav",
        ),
        (
            "Slow & clear narration",
            "Please speak slowly and very clearly, as in an audio book.<|endofprompt|>",
            "output_instruct_slow_clear.wav",
        ),
    ]

    for label, instruct_text, filename in instruct_configs:
        run_synthesis(
            f"Instruct: {label}",
            lambda it=instruct_text: model.inference_instruct2(
                tts_text, it, ref_path, stream=False
            ),
            str(output_dir / filename),
            model.sample_rate,
        )


def demo_fine_grained_control(model, output_dir: Path) -> None:
    """
    Fine-grained prosody control using inline markup tags:
      [laughter]  — insert laughter
      [breath]    — insert a breath pause
      <strong>…</strong> is NOT supported in CosyVoice2 (only CosyVoice1).
    Uses reference.wav for voice identity via cross-lingual inference.
    """
    print_banner("FINE-GRAINED CONTROL  (laughter / breath markup)")

    if not REFERENCE_WAV.exists():
        print("  ⚠  reference.wav not found — skipping fine-grained demo.")
        return

    ref_path = str(REFERENCE_WAV)

    # Text with prosody markup
    marked_text = (
        "Welcome to the demo. [breath] "
        "We are about to witness something truly remarkable. [laughter] "
        "The model can even insert natural pauses and emotional sounds [breath] "
        "right in the middle of synthesis."
    )

    run_synthesis(
        "Fine-grained control  [laughter + breath]",
        lambda: model.inference_cross_lingual(
            marked_text, ref_path, stream=False
        ),
        str(output_dir / "output_fine_grained.wav"),
        model.sample_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  CosyVoice 2-0.5B  ─  Quick-Look Evaluation Demo")
    print("  Model  : FunAudioLLM/CosyVoice2-0.5B")
    print("  Device : MPS (Apple Silicon) / CPU  [no CUDA on macOS]")
    print("=" * 72)

    # ── Read input text ───────────────────────────────────────────────────────
    if INPUT_TEXT_FILE.exists():
        tts_text = INPUT_TEXT_FILE.read_text(encoding="utf-8").strip()
        print(f"\n  📄 Input text loaded from {INPUT_TEXT_FILE.name} ({len(tts_text)} chars)")
    else:
        tts_text = (
            "Welcome to the CosyVoice 2 quick-look demo. "
            "This model supports zero-shot voice cloning and multi-style synthesis."
        )
        print(f"\n  📄 Using default input text ({len(tts_text)} chars)")

    print(f"  Text preview: \"{tts_text[:80]}{'…' if len(tts_text) > 80 else ''}\"")

    # ── Set up paths ──────────────────────────────────────────────────────────
    if not setup_paths():
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  📁 Output directory: {OUTPUT_DIR}")

    # ── Load model ────────────────────────────────────────────────────────────
    print_banner("MODEL LOADING")
    t_load_start = time.perf_counter()
    try:
        model = load_model()
    except Exception as e:
        print(f"\n  ✗  Failed to load model: {e}")
        log.debug("Model load error", exc_info=True)
        sys.exit(1)

    if model is None:
        sys.exit(1)

    load_time = time.perf_counter() - t_load_start
    print(f"  ✓  Model ready in {load_time:.1f}s  |  sample_rate={model.sample_rate} Hz")

    # ── Run all synthesis demos ───────────────────────────────────────────────
    demo_zero_shot_cloning(model, tts_text, OUTPUT_DIR)
    demo_instruct_modes(model, tts_text, OUTPUT_DIR)
    demo_fine_grained_control(model, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_banner("SUMMARY")
    wav_files = sorted(OUTPUT_DIR.glob("output_*.wav"))
    if wav_files:
        print(f"  Generated {len(wav_files)} audio file(s) in {OUTPUT_DIR.name}/:\n")
        for f in wav_files:
            dur = get_wav_duration_seconds(str(f))
            size_kb = f.stat().st_size / 1024
            print(f"    {'✓':2} {f.name:<42}  {dur:5.1f}s  ({size_kb:.0f} KB)")
    else:
        print("  No audio files were generated.")
        print("  Check that reference.wav exists and the model downloaded correctly.")

    print(f"\n  Total wall-clock time: {time.perf_counter() - t_load_start:.1f}s")
    print("\n  ✅ Demo complete. Open the output_*.wav files to evaluate audio quality.\n")


if __name__ == "__main__":
    main()
