"""Central configuration for realtime mel-spectrogram app."""

import pyaudio

# Audio input settings
PYAUDIO_FORMAT: int = pyaudio.paFloat32
INPUT_DEVICE_KEYWORD: str = "VoiceMeeter Output"
MAX_INPUT_CHANNELS: int = 2
CHUNK: int = 1024

# Mel-spectrogram and viewer settings
N_MELS: int = 128
N_FRAMES: int = 50
FPS: int = 30
WINDOW_SIZE: tuple[int, int] = (1000, 1000)
TITLE: str = "realtime melspectrogram"
COLORMAP_NAME: str = "magma"
