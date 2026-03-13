"""
ComfyUI custom nodes: Spectrogram → Audio (Phase Retrieval)
Supports linear and mel spectrograms via Griffin-Lim algorithm.
"""

import os
import torch
import numpy as np
import folder_paths

try:
    import librosa
    import soundfile as sf
except ImportError:
    raise ImportError("librosa and soundfile are required. Run: pip install librosa soundfile")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _image_to_magnitude(spectrogram_tensor: torch.Tensor, scale: str, db_range: float) -> np.ndarray:
    """
    Convert a ComfyUI IMAGE tensor [B, H, W, C] → 2-D magnitude array [freq, time].

    Supported scale values
    ----------------------
    'log_db'      : image is a normalized dB spectrogram  (0→-db_range dB, 1→0 dB)
    'log_natural' : image was log(magnitude+1) normalized to [0, 1]
    'linear'      : image pixel = magnitude (already linear, just rescaled to [0,1])
    """
    # Use the first batch item, convert to grayscale by averaging channels
    img = spectrogram_tensor[0].cpu().numpy()          # [H, W, C]
    if img.shape[2] == 3 or img.shape[2] == 4:
        gray = img[..., :3].mean(axis=2)               # [H, W]
    else:
        gray = img[..., 0]

    # Spectrogram convention: low freq at bottom → flip vertical axis
    gray = np.flipud(gray)                             # [freq, time]

    if scale == "log_db":
        # gray ∈ [0,1] → dB ∈ [-db_range, 0] → linear amplitude
        db = gray * db_range - db_range                # [-db_range, 0]
        magnitude = librosa.db_to_amplitude(db, ref=1.0)
    elif scale == "log_natural":
        # gray ∈ [0,1] represents log1p(magnitude) normalized
        magnitude = np.expm1(gray * np.log1p(1.0))    # rough inversion
    else:  # linear
        magnitude = gray

    return magnitude.astype(np.float32)


def _run_griffin_lim(magnitude: np.ndarray, n_fft: int, hop_length: int,
                     win_length: int, n_iter: int, power: float) -> np.ndarray:
    """Run Griffin-Lim phase retrieval and return mono waveform."""
    mag = magnitude ** power
    audio = librosa.griffinlim(
        mag,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window="hann",
        center=True,
        dtype=np.float32,
        momentum=0.99,
    )
    return audio


def _waveform_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    """Pack a mono numpy waveform into ComfyUI AUDIO format {waveform, sample_rate}."""
    # ComfyUI AUDIO: waveform tensor [B, C, T]
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,T]
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Node 1 – Linear Spectrogram → Audio
# ---------------------------------------------------------------------------

class LinearSpectrogramToAudio:
    """
    Converts a linear or log-magnitude spectrogram image to audio
    using the Griffin-Lim phase retrieval algorithm.

    Input IMAGE is expected to be a spectrogram with:
      - Horizontal axis = time
      - Vertical axis   = frequency (low freq at bottom)
    """

    SCALE_OPTIONS = ["log_db", "log_natural", "linear"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spectrogram": ("IMAGE",),
                "sample_rate":  ("INT",   {"default": 22050, "min": 8000,  "max": 48000, "step": 1}),
                "n_fft":        ("INT",   {"default": 2048,  "min": 64,    "max": 8192,  "step": 64}),
                "hop_length":   ("INT",   {"default": 512,   "min": 32,    "max": 2048,  "step": 32}),
                "win_length":   ("INT",   {"default": 2048,  "min": 64,    "max": 8192,  "step": 64}),
                "n_iter":       ("INT",   {"default": 60,    "min": 1,     "max": 512,   "step": 1}),
                "power":        ("FLOAT", {"default": 1.0,   "min": 0.1,   "max": 3.0,   "step": 0.1}),
                "scale":        (cls.SCALE_OPTIONS, {"default": "log_db"}),
                "db_range":     ("FLOAT", {"default": 80.0,  "min": 20.0,  "max": 160.0, "step": 10.0}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "convert"
    CATEGORY      = "audio/spectrogram"

    def convert(self, spectrogram, sample_rate, n_fft, hop_length, win_length,
                n_iter, power, scale, db_range):
        magnitude = _image_to_magnitude(spectrogram, scale, db_range)

        # Validate shape: should be [n_fft//2+1, time_frames]
        expected_freq_bins = n_fft // 2 + 1
        if magnitude.shape[0] != expected_freq_bins:
            magnitude = librosa.util.fix_length(
                magnitude,
                size=expected_freq_bins,
                axis=0,
            )

        audio_np = _run_griffin_lim(magnitude, n_fft, hop_length, win_length, n_iter, power)
        return (_waveform_to_comfy(audio_np, sample_rate),)


# ---------------------------------------------------------------------------
# Node 2 – Mel Spectrogram → Audio
# ---------------------------------------------------------------------------

class MelSpectrogramToAudio:
    """
    Converts a Mel spectrogram image to audio:
      1. Inverse Mel filterbank  (mel → linear magnitude)
      2. Griffin-Lim phase retrieval  (magnitude → waveform)
    """

    SCALE_OPTIONS = ["log_db", "log_natural", "linear"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mel_spectrogram": ("IMAGE",),
                "sample_rate":  ("INT",   {"default": 22050, "min": 8000,  "max": 48000, "step": 1}),
                "n_fft":        ("INT",   {"default": 2048,  "min": 64,    "max": 8192,  "step": 64}),
                "hop_length":   ("INT",   {"default": 512,   "min": 32,    "max": 2048,  "step": 32}),
                "win_length":   ("INT",   {"default": 2048,  "min": 64,    "max": 8192,  "step": 64}),
                "n_mels":       ("INT",   {"default": 128,   "min": 16,    "max": 512,   "step": 8}),
                "fmin":         ("FLOAT", {"default": 0.0,   "min": 0.0,   "max": 8000.0,"step": 10.0}),
                "fmax":         ("FLOAT", {"default": 8000.0,"min": 100.0, "max": 24000.0,"step": 100.0}),
                "n_iter":       ("INT",   {"default": 60,    "min": 1,     "max": 512,   "step": 1}),
                "power":        ("FLOAT", {"default": 1.0,   "min": 0.1,   "max": 3.0,   "step": 0.1}),
                "scale":        (cls.SCALE_OPTIONS, {"default": "log_db"}),
                "db_range":     ("FLOAT", {"default": 80.0,  "min": 20.0,  "max": 160.0, "step": 10.0}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "convert"
    CATEGORY      = "audio/spectrogram"

    def convert(self, mel_spectrogram, sample_rate, n_fft, hop_length, win_length,
                n_mels, fmin, fmax, n_iter, power, scale, db_range):
        mel_mag = _image_to_magnitude(mel_spectrogram, scale, db_range)

        # Resize to n_mels if the image height doesn't match
        if mel_mag.shape[0] != n_mels:
            mel_mag = librosa.util.fix_length(mel_mag, size=n_mels, axis=0)

        # Inverse Mel filterbank: mel magnitude → linear magnitude
        linear_mag = librosa.feature.inverse.mel_to_stft(
            mel_mag,
            sr=sample_rate,
            n_fft=n_fft,
            power=power,
            fmin=fmin,
            fmax=fmax if fmax > 0 else None,
        )

        audio_np = _run_griffin_lim(linear_mag, n_fft, hop_length, win_length, n_iter, power=1.0)
        return (_waveform_to_comfy(audio_np, sample_rate),)


# ---------------------------------------------------------------------------
# Node 3 – Save Audio
# ---------------------------------------------------------------------------

class SaveAudio:
    """Save AUDIO to a .wav or .flac file in ComfyUI's output directory."""

    FORMAT_OPTIONS = ["wav", "flac"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":    ("AUDIO",),
                "filename": ("STRING", {"default": "output_audio"}),
                "format":   (cls.FORMAT_OPTIONS, {"default": "wav"}),
            }
        }

    RETURN_TYPES  = ()
    OUTPUT_NODE   = True
    FUNCTION      = "save"
    CATEGORY      = "audio/spectrogram"

    def save(self, audio, filename, format):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        waveform   = audio["waveform"]   # [B, C, T]
        sample_rate = audio["sample_rate"]

        # Flatten batch & channels → mono
        wav_np = waveform[0].mean(dim=0).cpu().numpy()  # [T]

        # Avoid clipping
        peak = np.abs(wav_np).max()
        if peak > 1.0:
            wav_np = wav_np / peak

        out_path = os.path.join(output_dir, f"{filename}.{format}")
        sf.write(out_path, wav_np, sample_rate, subtype="PCM_16")
        print(f"[spectrogram2audio] Saved → {out_path}")
        return {}


# ---------------------------------------------------------------------------
# Node 4 – Preview Audio Info (utility)
# ---------------------------------------------------------------------------

class AudioInfo:
    """Display basic info about an AUDIO tensor (duration, sample rate, channels)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("info",)
    FUNCTION      = "inspect"
    CATEGORY      = "audio/spectrogram"

    def inspect(self, audio):
        waveform    = audio["waveform"]       # [B, C, T]
        sample_rate = audio["sample_rate"]
        B, C, T     = waveform.shape
        duration    = T / sample_rate
        info = (
            f"Batch:       {B}\n"
            f"Channels:    {C}\n"
            f"Samples:     {T}\n"
            f"Sample rate: {sample_rate} Hz\n"
            f"Duration:    {duration:.3f} s"
        )
        print(f"[spectrogram2audio] {info}")
        return (info,)
