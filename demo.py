"""
demo.py — CosyVoice2-0.5B Human-Like TTS Demo
================================================
Generates a natural, conversational English audio clip using CosyVoice2's
most expressive inference mode: inference_instruct2.

Prerequisites:
    1. pip install -r requirements_macos.txt   (done once)
    2. python download_model.py                (done once, ~2 GB)

Run:
    python demo.py

Output:
    output_demo.wav  — playable WAV file in the repo root
"""

import sys
import os

# CosyVoice source lives in cosyvoice_src/ (cloned from FunAudioLLM/CosyVoice)
_ROOT = os.path.dirname(__file__)
_SRC  = os.path.join(_ROOT, "cosyvoice_src")

# Add cosyvoice package and its Matcha-TTS submodule to path
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "third_party", "Matcha-TTS"))

# Also change working directory to cosyvoice_src so relative model paths resolve
os.chdir(_SRC)

import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

# ─────────────────────────────────────────────
# 1. Load the model
#    load_jit=False  → disables TorchScript JIT  (Linux optimisation, not needed on macOS)
#    load_trt=False  → disables TensorRT          (NVIDIA/CUDA only, not available on macOS)
#    fp16=False      → disables half-precision    (not stable on CPU/MPS)
# ─────────────────────────────────────────────
print("Loading CosyVoice2-0.5B model (this takes ~30-60 seconds on first load)...")
cosyvoice = CosyVoice2(
    os.path.join(_ROOT, "pretrained_models", "CosyVoice2-0.5B"),
    load_jit=False,
    load_trt=False,
    fp16=False,
)
print(f"✅ Model loaded. Sample rate: {cosyvoice.sample_rate} Hz\n")

# ─────────────────────────────────────────────
# 2. Text to synthesise
#    Crafted for natural prosody:
#      - Conversational, human-paced sentence structure
#      - Em-dashes and ellipsis introduce natural pauses
#      - Enthusiastic but measured tone
# ─────────────────────────────────────────────
tts_text = ("""Concerns are rising about other nations using AI to remove humans from decision-making. The Pentagon moved to ban certain providers like Anthropic earlier this year... due to supply chain risks. This prompted a discussion with Under Secretary Emil Michael about how AI might change warfare... He draws an analogy to the ride-sharing industry. Safety statistics for systems like Tesla FSD are actually amazing. The fear is of change itself... in reality, the technology makes service more reliable and precise. Much like how Uber reduced drinking driving while increasing availability... applying this to military contexts means being able to discern a decoy from a non-decoy within drone swarms. A recent demonstration by Cameron Stanley... showcased this through a program called Maven Smart System. Specifically, the Target Workbench allows users to view live images and select targets within a unified workflow. It is not about replacing human judgment... but increasing the "human context window" by synthesizing vast amounts of data.

When a target is identified, the system calculates variables such as weather conditions... fuel consumption, and collateral effects. It does not operate like a chatbot or Skynet; instead, it serves as an orchestration layer on top of data streams. The Under Secretary clarifies that no Large Language Model is baked into the kill chain itself... countering common misconceptions about automated killing. Instead, tools like Palantir surface choices that are otherwise consumed by spreadsheets and PowerPoint files—methods historically used to relay target lists. The digitalization of targeting processes accelerates these decisions, granting a single operator the power of many more. While permissions and authorities remain strictly human-controlled to ensure checks and balances, the system provides better outcomes through informed clicks. This shift from manual coordination to AI-assisted synthesis represents a responsible evolution of war fighting... moving beyond the chaos of unconnected data to a unified strategy.

The discussion outlines three layers of artificial intelligence application within defense, starting with efficiency... Mundane work is streamlined so personnel can focus on more interesting tasks. Then there is the intelligence layer... Imagine all the intelligence gathered from satellite imagery worldwide. Currently, a human analyst must look at everything to make a judgment... but with historical data and AI synthesis, the system can identify anomalies. It learns what those anomalies are—creating a totally different paradigm for intelligence analysis if you will. Moving on to the third layer, war fighting... AI takes all paperwork and modeling and simulation to react faster. But also more precisely.
Moving to conflicts of interest, the host queried about XAI and SpaceX holdings. The official confirmed he sold all SpaceX stock to comply with the Office of Government Ethics... Defense company stocks are red lines, and he recused himself from XAI dealings until the sale cleared. Next came procurement reform involving Uber and Palantir dynamics... Defense contractors consolidated from fifty down to five since the Cold War, making supply chains brittle.

The official argued shifting from cost-plus contracts to performance-based deals is necessary... If a weapon works on time, they get paid; if not, they do not. This risk-sharing model benefits taxpayers and encourages innovation without massive speculative R&D burdens... Founders like Palmer Lucky are willing to enter this business finally. Finally, the conversation addressed the Pentagon Pizza Index tracking orders to predict military action... The official dismissed this entirely stating he has no idea how food gets delivered inside the Pentagon building. There are no specific Papa John's locations delivering directly in... The segment concluded by thanking guests for visiting Washington DC. This occurred before signing off on Big Technology Podcast.""")

# ─────────────────────────────────────────────
# 3. Instruction prompt
#    Natural language style directive embedded with CosyVoice2's special tag.
#    Controls: warmth, pacing, emotional register, conversational feel.
# ─────────────────────────────────────────────
instruct_text = (
    "Speak in a warm, clear, conversational tone with natural pacing, "
    "slight enthusiasm, and human-like breathing rhythms."
    "<|endofprompt|>"
)

# ─────────────────────────────────────────────
# 4. Reference voice (prompt audio)
#    Uses the built-in sample that ships with the CosyVoice repo.
#    This is a ~3-second neutral voice clip used for zero-shot cloning.
# ─────────────────────────────────────────────
prompt_wav_path = os.path.join(_SRC, "asset", "zero_shot_prompt.wav")

if not os.path.exists(prompt_wav_path):
    raise FileNotFoundError(
        f"Reference audio not found at: {prompt_wav_path}\n"
        "Make sure you cloned the full CosyVoice repo (including the 'asset/' folder)."
    )

print(f"Using reference voice: {prompt_wav_path}")

# ─────────────────────────────────────────────
# 5. Run inference
#    inference_instruct2 = most expressive mode:
#      combines zero-shot voice cloning + natural language style instructions.
#    prompt_wav_path: pass file path (string) — CosyVoice loads it internally at 24kHz
#    stream=False → generate the full audio at once (simpler for demo)
# ─────────────────────────────────────────────
output_file = os.path.join(_ROOT, "output_demo.wav")

print("\nGenerating speech (may take 30-120 seconds on CPU)...")
print(f"Text: \"{tts_text[:80]}...\"\n")

generated = False
for i, result in enumerate(cosyvoice.inference_instruct2(
    tts_text,
    instruct_text,
    prompt_wav_path,
    stream=False,
)):
    audio_tensor = result["tts_speech"]
    torchaudio.save(output_file, audio_tensor, cosyvoice.sample_rate)
    duration = audio_tensor.shape[-1] / cosyvoice.sample_rate
    print(f"✅ Chunk {i}: {duration:.2f}s saved → {output_file}")
    generated = True

if generated:
    print(f"\n🎧 Done! Play your audio: open {output_file}")
    print("   (Or double-click the file in Finder to open with QuickTime)")
else:
    print("⚠️  No audio was generated. Check the model and inputs.")
