"""Entry point to run realtime mel-spectrogram viewer."""

import threading

from .audio_stream import AudioInputStream
from .melspectrogram_viewer import MelSpectrogramViewer


def main() -> None:
    """Start audio capture in a worker thread and launch the viewer UI."""
    # Get PyAudio stream input
    ais: AudioInputStream = AudioInputStream(CHUNK=1024)
    melspectrogram: MelSpectrogramViewer = MelSpectrogramViewer(
        ais.RATE, (ais.CHANNELS, ais.CHUNK)
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
