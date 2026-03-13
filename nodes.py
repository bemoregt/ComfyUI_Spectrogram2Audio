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
    from scipy.ndimage import zoom
except ImportError:
    raise ImportError(
        "librosa, soundfile, and scipy are required. "
        "Run: pip install librosa soundfile scipy"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _image_to_magnitude(
    spectrogram_tensor: torch.Tensor,
    scale: str,
    db_range: float,
    flip_freq: bool,
) -> np.ndarray:
    """
    Convert a ComfyUI IMAGE tensor [B, H, W, C] → 2-D magnitude array [freq, time].

    scale options
    -------------
    'amplitude_db'  : pixels encode dB of an amplitude spectrogram
                      (librosa.amplitude_to_db output, 20*log10 scale)
    'power_db'      : pixels encode dB of a power spectrogram
                      (librosa.power_to_db output, 10*log10 scale)
    'log_natural'   : pixels = log1p(magnitude), normalized to [0, 1]
    'linear'        : pixels ∝ magnitude (linear scale)
    """
    img = spectrogram_tensor[0].cpu().numpy()   # [H, W, C]

    # Luminance-weighted grayscale (more accurate than mean for colormaps)
    if img.shape[2] >= 3:
        gray = (0.299 * img[..., 0]
                + 0.587 * img[..., 1]
                + 0.114 * img[..., 2])          # [H, W]
    else:
        gray = img[..., 0]

    if flip_freq:
        # librosa stores freq bin 0 (DC) at row 0 of the array.
        # Standard spectrogram images have low freq at the *bottom* of the
        # display, which means row 0 of the image = high freq → need flip.
        gray = np.flipud(gray)                  # [freq, time]

    if scale == "amplitude_db":
        # pixel ∈ [0,1] → dB ∈ [-db_range, 0] → amplitude
        db = gray * db_range - db_range
        magnitude = librosa.db_to_amplitude(db, ref=1.0)

    elif scale == "power_db":
        # pixel ∈ [0,1] → dB ∈ [-db_range, 0] → power → amplitude
        db = gray * db_range - db_range
        power_mag = librosa.db_to_power(db, ref=1.0)
        magnitude = np.sqrt(power_mag)          # convert power → amplitude

    elif scale == "log_natural":
        # pixel = log1p(magnitude) / log1p(max_val), max_val assumed = 1
        magnitude = np.expm1(gray)

    else:  # "linear"
        magnitude = gray

    return magnitude.astype(np.float32)


def _resize_freq_axis(magnitude: np.ndarray, target_bins: int) -> np.ndarray:
    """Resize the frequency axis via bilinear interpolation (no zero-padding)."""
    if magnitude.shape[0] == target_bins:
        return magnitude
    ratio = target_bins / magnitude.shape[0]
    return zoom(magnitude, (ratio, 1.0), order=1).astype(np.float32)


def _run_griffin_lim(
    magnitude: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    power: float,
) -> np.ndarray:
    """
    Run Griffin-Lim phase retrieval and return a mono waveform.

    `magnitude` must be an *amplitude* spectrogram coming in.
    `power` describes what the original spectrogram represents:
        power=1.0  →  amplitude spectrogram  (no conversion needed)
        power=2.0  →  power spectrogram      (take sqrt before GLA)
    We apply magnitude ** (1/power) to convert to amplitude before GLA.
    """
    amp = magnitude ** (1.0 / power)   # ← was `magnitude ** power` (wrong)
    audio = librosa.griffinlim(
        amp,
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
    """Pack a mono numpy waveform into ComfyUI AUDIO format."""
    # ComfyUI AUDIO: waveform tensor [B, C, T]
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Node 1 – Linear Spectrogram → Audio
# ---------------------------------------------------------------------------

class LinearSpectrogramToAudio:
    """
    Converts a linear or log-magnitude STFT spectrogram image to audio
    using the Griffin-Lim phase retrieval algorithm.

    Input IMAGE convention (default):
      - Horizontal axis = time
      - Vertical axis   = frequency, low freq at the bottom of the image
    """

    SCALE_OPTIONS = ["amplitude_db", "power_db", "log_natural", "linear"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spectrogram":  ("IMAGE",),
                "sample_rate":  ("INT",     {"default": 22050, "min": 8000,  "max": 48000,  "step": 1}),
                "n_fft":        ("INT",     {"default": 2048,  "min": 64,    "max": 8192,   "step": 64}),
                "hop_length":   ("INT",     {"default": 512,   "min": 32,    "max": 2048,   "step": 32}),
                "win_length":   ("INT",     {"default": 2048,  "min": 64,    "max": 8192,   "step": 64}),
                "n_iter":       ("INT",     {"default": 60,    "min": 1,     "max": 512,    "step": 1}),
                "power":        ("FLOAT",   {"default": 1.0,   "min": 1.0,   "max": 2.0,    "step": 1.0,
                                             "tooltip": "1.0=amplitude spectrogram, 2.0=power spectrogram"}),
                "scale":        (cls.SCALE_OPTIONS, {"default": "amplitude_db"}),
                "db_range":     ("FLOAT",   {"default": 80.0,  "min": 20.0,  "max": 160.0,  "step": 10.0}),
                "flip_freq":    ("BOOLEAN", {"default": True,
                                             "tooltip": "Flip frequency axis. True if low freq is at bottom of image"}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "convert"
    CATEGORY      = "audio/spectrogram"

    def convert(self, spectrogram, sample_rate, n_fft, hop_length, win_length,
                n_iter, power, scale, db_range, flip_freq):
        magnitude = _image_to_magnitude(spectrogram, scale, db_range, flip_freq)

        # Resize frequency axis via interpolation to match n_fft
        expected_bins = n_fft // 2 + 1
        magnitude = _resize_freq_axis(magnitude, expected_bins)

        audio_np = _run_griffin_lim(magnitude, n_fft, hop_length, win_length, n_iter, power)
        return (_waveform_to_comfy(audio_np, sample_rate),)


# ---------------------------------------------------------------------------
# Node 2 – Mel Spectrogram → Audio
# ---------------------------------------------------------------------------

class MelSpectrogramToAudio:
    """
    Converts a Mel spectrogram image to audio:
      1. Inverse Mel filterbank  (mel → linear STFT magnitude)
      2. Griffin-Lim phase retrieval  (magnitude → waveform)
    """

    SCALE_OPTIONS = ["amplitude_db", "power_db", "log_natural", "linear"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mel_spectrogram": ("IMAGE",),
                "sample_rate":  ("INT",     {"default": 22050, "min": 8000,  "max": 48000,  "step": 1}),
                "n_fft":        ("INT",     {"default": 2048,  "min": 64,    "max": 8192,   "step": 64}),
                "hop_length":   ("INT",     {"default": 512,   "min": 32,    "max": 2048,   "step": 32}),
                "win_length":   ("INT",     {"default": 2048,  "min": 64,    "max": 8192,   "step": 64}),
                "n_mels":       ("INT",     {"default": 128,   "min": 16,    "max": 512,    "step": 8}),
                "fmin":         ("FLOAT",   {"default": 0.0,   "min": 0.0,   "max": 8000.0, "step": 10.0}),
                "fmax":         ("FLOAT",   {"default": 8000.0,"min": 100.0, "max": 24000.0,"step": 100.0}),
                "n_iter":       ("INT",     {"default": 60,    "min": 1,     "max": 512,    "step": 1}),
                "power":        ("FLOAT",   {"default": 2.0,   "min": 1.0,   "max": 2.0,    "step": 1.0,
                                             "tooltip": "1.0=amplitude mel, 2.0=power mel (librosa default)"}),
                "scale":        (cls.SCALE_OPTIONS, {"default": "power_db"}),
                "db_range":     ("FLOAT",   {"default": 80.0,  "min": 20.0,  "max": 160.0,  "step": 10.0}),
                "flip_freq":    ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "convert"
    CATEGORY      = "audio/spectrogram"

    def convert(self, mel_spectrogram, sample_rate, n_fft, hop_length, win_length,
                n_mels, fmin, fmax, n_iter, power, scale, db_range, flip_freq):
        mel_mag = _image_to_magnitude(mel_spectrogram, scale, db_range, flip_freq)

        # Resize to n_mels via interpolation if image height doesn't match
        mel_mag = _resize_freq_axis(mel_mag, n_mels)

        # Inverse Mel filterbank → linear amplitude STFT
        # librosa.mel_to_stft expects the raw mel spectrogram (not yet in amplitude),
        # and `power` tells it whether it's amplitude (1) or power (2) mel.
        # It returns an amplitude STFT in both cases.
        linear_amp = librosa.feature.inverse.mel_to_stft(
            mel_mag,
            sr=sample_rate,
            n_fft=n_fft,
            power=power,
            fmin=fmin,
            fmax=fmax,
        )

        # linear_amp is already amplitude STFT → pass power=1.0 to GLA
        audio_np = _run_griffin_lim(linear_amp, n_fft, hop_length, win_length, n_iter, power=1.0)
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

        waveform    = audio["waveform"]     # [B, C, T]
        sample_rate = audio["sample_rate"]

        wav_np = waveform[0].mean(dim=0).cpu().numpy()  # mono [T]

        peak = np.abs(wav_np).max()
        if peak > 1.0:
            wav_np = wav_np / peak

        out_path = os.path.join(output_dir, f"{filename}.{format}")
        sf.write(out_path, wav_np, sample_rate, subtype="PCM_16")
        print(f"[spectrogram2audio] Saved → {out_path}")
        return {}


# ---------------------------------------------------------------------------
# Node 4 – Audio Info
# ---------------------------------------------------------------------------

class AudioInfo:
    """Display basic info about an AUDIO tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("info",)
    FUNCTION      = "inspect"
    CATEGORY      = "audio/spectrogram"

    def inspect(self, audio):
        waveform    = audio["waveform"]
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
