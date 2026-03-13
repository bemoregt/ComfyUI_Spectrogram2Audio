# ComfyUI Spectrogram → Audio

A ComfyUI custom node package that reconstructs audio from spectrogram images using **phase retrieval** (Griffin-Lim algorithm). Supports both linear magnitude spectrograms and Mel spectrograms.

---

## Nodes

### Linear Spectrogram → Audio (Griffin-Lim)
**Category:** `audio/spectrogram`

Converts a linear or log-magnitude STFT spectrogram image directly to audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectrogram` | IMAGE | — | Input spectrogram (freq on vertical axis, time on horizontal) |
| `sample_rate` | INT | 22050 | Output audio sample rate (Hz) |
| `n_fft` | INT | 2048 | FFT window size |
| `hop_length` | INT | 512 | STFT hop length (samples) |
| `win_length` | INT | 2048 | STFT window length (samples) |
| `n_iter` | INT | 60 | Griffin-Lim iterations — more = better quality, slower |
| `power` | FLOAT | 1.0 | Exponent applied to magnitude before inversion (1.0 = amplitude, 2.0 = power) |
| `scale` | ENUM | log_db | Pixel-to-magnitude mapping (see [Scale Modes](#scale-modes)) |
| `db_range` | FLOAT | 80.0 | Dynamic range in dB (used only when `scale = log_db`) |

**Output:** `AUDIO`

---

### Mel Spectrogram → Audio (Griffin-Lim)
**Category:** `audio/spectrogram`

Converts a Mel spectrogram image to audio via two steps:
1. **Inverse Mel filterbank** — maps Mel bins back to linear STFT bins
2. **Griffin-Lim** — recovers phase and synthesizes audio

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mel_spectrogram` | IMAGE | — | Input Mel spectrogram image |
| `sample_rate` | INT | 22050 | Output audio sample rate (Hz) |
| `n_fft` | INT | 2048 | FFT window size |
| `hop_length` | INT | 512 | STFT hop length (samples) |
| `win_length` | INT | 2048 | STFT window length (samples) |
| `n_mels` | INT | 128 | Number of Mel filterbank bins |
| `fmin` | FLOAT | 0.0 | Lowest frequency for Mel filterbank (Hz) |
| `fmax` | FLOAT | 8000.0 | Highest frequency for Mel filterbank (Hz) |
| `n_iter` | INT | 60 | Griffin-Lim iterations |
| `power` | FLOAT | 1.0 | Magnitude exponent |
| `scale` | ENUM | log_db | Pixel-to-magnitude mapping |
| `db_range` | FLOAT | 80.0 | Dynamic range in dB |

**Output:** `AUDIO`

---

### Save Audio
**Category:** `audio/spectrogram`

Saves an `AUDIO` tensor to a file in ComfyUI's output directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | — | Audio to save |
| `filename` | STRING | output_audio | Output filename (without extension) |
| `format` | ENUM | wav | File format: `wav` or `flac` |

---

### Audio Info
**Category:** `audio/spectrogram`

Displays metadata of an `AUDIO` tensor (batch size, channels, sample count, sample rate, duration).

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | AUDIO | Audio to inspect |

**Output:** `STRING` — formatted info text

---

## Scale Modes

The `scale` parameter controls how image pixel values (0–1) are interpreted as magnitude values.

| Mode | Formula | When to use |
|------|---------|-------------|
| `log_db` | `pixel → dB ∈ [−db_range, 0] → amplitude` | Most visualization tools (e.g. librosa's `specshow`, Audacity exports) |
| `log_natural` | `pixel → expm1(pixel × log(2))` | Spectrograms saved with `log1p` normalization |
| `linear` | `pixel = magnitude` | Raw magnitude arrays saved directly as images |

> **Tip:** If you generated the spectrogram with `librosa.amplitude_to_db()` or similar, use `log_db`.

---

## Algorithm — Griffin-Lim Phase Retrieval

Only the **magnitude** of a spectrogram is stored in an image; the **phase** is lost. Griffin-Lim reconstructs the phase iteratively:

```
Initialize: random phase θ₀
Loop n_iter times:
    S  = magnitude ⊙ exp(iθ)      ← impose known magnitude
    s  = iSTFT(S)                  ← synthesize waveform
    S' = STFT(s)                   ← re-analyze
    θ  = angle(S')                 ← update phase estimate
Return: iSTFT(magnitude ⊙ exp(iθ))
```

This implementation uses `momentum=0.99` (Fast Griffin-Lim variant), which converges faster than the original algorithm.

**Quality tips:**
- Increase `n_iter` (60 → 128+) for cleaner results at the cost of speed.
- Match `n_fft`, `hop_length`, `win_length`, and `sample_rate` exactly to the parameters used when the spectrogram was originally created.
- For Mel spectrograms, also match `n_mels`, `fmin`, and `fmax`.

---

## Installation

```bash
# 1. Clone into your ComfyUI custom_nodes directory
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI_spectrogram2audio

# 2. Install dependencies
pip install -r ComfyUI_spectrogram2audio/requirements.txt

# 3. Restart ComfyUI
```

### Requirements

```
librosa >= 0.10.0
soundfile >= 0.12.0
numpy >= 1.24.0
```

---

## Example Workflows

### Linear spectrogram → WAV

```
[Load Image] ──► [Linear Spectrogram → Audio]
                      sample_rate: 22050
                      n_fft:       2048
                      hop_length:  512
                      n_iter:      60
                      scale:       log_db
                      db_range:    80
                 └──► [Save Audio]
                           filename: my_audio
                           format:   wav
```

### Mel spectrogram → WAV

```
[Load Image] ──► [Mel Spectrogram → Audio]
                      sample_rate: 22050
                      n_fft:       2048
                      hop_length:  256
                      n_mels:      128
                      fmin:        0
                      fmax:        8000
                      n_iter:      80
                      scale:       log_db
                 └──► [Save Audio]
```

---

## Limitations

- Phase retrieval is an **approximation**. The reconstructed audio will sound similar to the original but is not identical, especially for complex signals.
- Griffin-Lim works best on **single-instrument or speech** audio. Mixed or highly transient signals may sound blurry.
- The spectrogram image must preserve the **correct aspect ratio** (frequency bins × time frames). Aggressive resizing will degrade quality.

---

## License

MIT
