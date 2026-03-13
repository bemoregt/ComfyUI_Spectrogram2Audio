from .nodes import (
    LinearSpectrogramToAudio,
    MelSpectrogramToAudio,
    SaveAudio,
    AudioInfo,
)

NODE_CLASS_MAPPINGS = {
    "LinearSpectrogramToAudio": LinearSpectrogramToAudio,
    "MelSpectrogramToAudio":    MelSpectrogramToAudio,
    "SaveAudio":                SaveAudio,
    "AudioInfo":                AudioInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearSpectrogramToAudio": "Linear Spectrogram → Audio (Griffin-Lim)",
    "MelSpectrogramToAudio":    "Mel Spectrogram → Audio (Griffin-Lim)",
    "SaveAudio":                "Save Audio",
    "AudioInfo":                "Audio Info",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
