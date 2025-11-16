# app_audio_web_en.py
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import gradio as gr

from src.models.cnn_small import SmallCNN   # your model class


def resolve_from_root(p: str) -> Path:
    """Resolve a path relative to the project root (folder that contains src/)."""
    p = Path(p)
    if p.is_absolute():
        return p
    proj_root = Path(__file__).resolve().parents[0]  # e.g. F:\asc
    return proj_root / p


# ===== 1. Load model and config =====
# Use the 9-class environmental model you just trained
CKPT_PATH = resolve_from_root("runs/custom_env_se/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ck = torch.load(str(CKPT_PATH), map_location="cpu")
cfg = ck["cfg"]
label2id = ck["label2id"]
id2label = {v: k for k, v in label2id.items()}
num_classes = len(label2id)

# Feature parameters (must match training)
SR = cfg["sr"]
N_FFT = cfg["n_fft"]
HOP = cfg["hop"]
N_MELS = cfg["n_mels"]
FMIN = cfg["fmin"]
FMAX = cfg["fmax"]
TARGET_FRAMES = cfg["target_frames"]

use_se = cfg.get("model", {}).get("use_se", True)
model = SmallCNN(num_classes, use_se=use_se)
model.load_state_dict(ck["model"], strict=True)
model.to(device).eval()

print("Loaded model with classes:", id2label)


# ===== 2. Feature extraction + inference for a single file =====
def melspec_from_path(path: str):
    """Compute log-Mel spectrogram [n_mels, T] from an audio file."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)

    # Convert to mono if necessary
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Ensure minimum length for one STFT frame
    if len(y) < N_FFT:
        pad = N_FFT - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    # Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    logS = librosa.power_to_db(S, ref=1.0)
    return logS  # [n_mels, T]


def classify_audio(file_path: str):
    """
    Gradio callback:
      input : path to an audio file
      output: {label: probability} dict, shown as a ranked label widget
    """
    if file_path is None or not os.path.isfile(file_path):
        return {"No file": 1.0}

    # 1) Extract log-Mel features
    feat = melspec_from_path(file_path)         # [n_mels, T]
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, T]

    # 2) Pad / crop to the target number of frames
    T = x.shape[-1]
    if T < TARGET_FRAMES:
        x = F.pad(x, (0, TARGET_FRAMES - T))
    else:
        x = x[..., :TARGET_FRAMES]

    x = x.to(device).float()

    # 3) Forward pass
    with torch.no_grad():
        logits = model(x)                       # [1, C]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # 4) Convert to {label: prob} for Gradio
    out = {id2label[i]: float(probs[i]) for i in range(len(id2label))}
    return out


# ===== 3. Build Gradio web UI (English) =====
description_md = """
### Environmental Sound Classifier (Demo)

This demo uses a convolutional neural network (SmallCNN with Squeeze-and-Excitation)
trained on environmental audio clips.

**Target classes (9):**
- Indoor: `indoor_speech`, `indoor_music`, `indoor_noise`, `other`
- Outdoor: `outdoor_rain`, `outdoor_thunder`, `outdoor_wind`,
  `outdoor_birdsong`, `outdoor_dog_bark`

**How to use:**
1. Drag and drop an audio file (preferably 1–10 seconds), or record via microphone.  
2. Click **Classify**.  
3. The right panel shows the top-5 predicted classes with probabilities.
"""

with gr.Blocks() as demo:
    gr.Markdown("# Environmental Sound Classifier")
    gr.Markdown(description_md)

    with gr.Row():
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Upload or record an audio clip",
        )
        label_out = gr.Label(
            num_top_classes=5,
            label="Predicted classes (Top-5)"
        )

    btn = gr.Button("Classify")

    btn.click(fn=classify_audio, inputs=audio_in, outputs=label_out)

    gr.Markdown(f"**Checkpoint used:** `{CKPT_PATH}`")

if __name__ == "__main__":
    # share=False for local use only; set True if you want a temporary public link
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)