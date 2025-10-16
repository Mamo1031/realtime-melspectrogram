"""Entry point to run realtime mel-spectrogram viewer."""

import threading

from .audio_stream import AudioInputStream
from .melspectrogram_viewer import MelSpectrogramViewer
from . import config


def main() -> None:
    """Start audio capture in a worker thread and launch the viewer UI."""
    # Get PyAudio stream input
    ais: AudioInputStream = AudioInputStream(
        CHUNK=config.CHUNK,
        format=config.PYAUDIO_FORMAT,
        input_device_keyword=config.INPUT_DEVICE_KEYWORD,
        maxInputChannels=config.MAX_INPUT_CHANNELS,
    )
    melspectrogram: MelSpectrogramViewer = MelSpectrogramViewer(
        ais.RATE,
        (ais.CHANNELS, ais.CHUNK),
        n_mels=config.N_MELS,
        n_frames=config.N_FRAMES,
        fps=config.FPS,
        size=config.WINDOW_SIZE,
        title=config.TITLE,
    )

    # Run AudioInputStream in a separate thread
    thread: threading.Thread = threading.Thread(
        target=ais.run, args=(melspectrogram.callback_sigproc,)
    )
    thread.daemon = True
    thread.start()

    # Start spectrogram drawing
    melspectrogram.run_app()


if __name__ == "__main__":
    main()
