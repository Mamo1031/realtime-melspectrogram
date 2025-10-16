"""Audio input streaming utility

It prints a compact device table on startup to help users find the right
device index and channel configuration.
"""

import pyaudio
import numpy as np
from typing import Callable, Dict, List, Optional


class AudioInputStream:
    """Continuous microphone capture with device selection and fixed-size frames.

    The class enumerates available input devices, chooses one either by
    explicit `input_device_index` or by fuzzy name match (`input_device_keyword`)
    and matching `maxInputChannels`, opens a PyAudio input stream, and passes
    deinterleaved frames to a user-provided callback.

    Attributes:
        maxInputChannels (int): Desired number of input channels to match
            when searching by keyword. Does not change the hardware capability.
        CHUNK (int): Number of frames per buffer (per channel) read from the stream.
        format (int): PyAudio sample format (e.g., ``pyaudio.paFloat32`` or ``paInt16``).
        RATE (int): Sample rate selected from the chosen device (fallback 44100).
        CHANNELS (int): Channel count selected from the chosen device (fallback 2).
        dtype (np.dtype): Numpy dtype corresponding to ``format`` for fast conversion.
        p (pyaudio.PyAudio): Underlying PyAudio instance.
        input_device_index (Optional[int]): Index of the selected input device.
        input_device_name (Optional[str]): Human-readable name of the selected device.
        devices (List[Dict]): Cached device info objects enumerated at open time.
        stream (pyaudio.Stream): Active input stream once opened.
    """

    def __init__(
        self,
        format: int = pyaudio.paFloat32,
        input_device_keyword: str = "VoiceMeeter Output",
        CHUNK: int = 1024,
        maxInputChannels: int = 2,
        input_device_index: Optional[int] = None,
    ) -> None:
        """Initialize the stream wrapper and open an input device.

        Args:
            format: PyAudio format (``pyaudio.paFloat32`` or ``pyaudio.paInt16``).
            input_device_keyword: Substring used to match a device name when
                ``input_device_index`` is not provided.
            CHUNK: Frames per buffer (per channel) for each read.
            maxInputChannels: Expected number of input channels to match against
                device capability when selecting by keyword.
            input_device_index: Explicit device index to force selection. If
                provided, ``input_device_keyword`` is ignored.

        Raises:
            ValueError: If an unsupported PyAudio format is specified.
        """
        self.maxInputChannels: int = maxInputChannels
        self.CHUNK: int = CHUNK
        self.format: int = format
        self.RATE: int = 44100  # Fallback default
        self.CHANNELS: int = 2  # Fallback default
        # Map PyAudio format to a NumPy dtype for zero-copy buffer views.
        if format is pyaudio.paFloat32:
            self.dtype = np.float32
        elif format is pyaudio.paInt16:
            self.dtype = np.int16
        else:
            raise ValueError("Unsupported PyAudio format. Use paFloat32 or paInt16.")
        # Create the PyAudio host and open the input stream.
        self.p = pyaudio.PyAudio()
        self.__open_stream(
            input_device_keyword=input_device_keyword,
            input_device_index=input_device_index,
        )

    def get_params(self) -> Dict[str, int]:
        """Return current runtime parameters.

        Returns:
            Dict[str, int]: A mapping with keys ``RATE``, ``CHUNK``, and ``CHANNELS``.
        """
        params_dict = {
            "RATE": self.RATE,
            "CHUNK": self.CHUNK,
            "CHANNELS": self.CHANNELS,
        }
        return params_dict

    @staticmethod
    def list_input_devices() -> List[Dict]:
        """Enumerate input/output audio devices visible to PyAudio.

        Returns:
            List[Dict]: Raw device info dictionaries as returned by PyAudio.

        Notes:
            This creates and terminates a temporary PyAudio instance to avoid
            interfering with an already-open stream.
        """
        p = pyaudio.PyAudio()
        devices: List[Dict] = []
        try:
            for k in range(p.get_device_count()):
                dev = p.get_device_info_by_index(k)
                devices.append(dev)
        finally:
            p.terminate()
        return devices

    def __open_stream(
        self, input_device_keyword: str, input_device_index: Optional[int]
    ) -> None:
        """Choose an input device and open the PyAudio input stream.

        Device selection strategy:
          1) If ``input_device_index`` is given and valid → select it.
          2) Else, find the first device whose name contains
             ``input_device_keyword`` **and** whose ``maxInputChannels`` equals
             ``self.maxInputChannels``.
          3) Else, fall back to the system default input device (best effort).

        Prints a device table to aid manual selection/debugging.
        """
        self.input_device_index: Optional[int] = None
        self.input_device_name: Optional[str] = None
        self.devices: List[Dict] = []
        print("=========================================================")
        print("dev. index\tmaxInputCh.\tmaxOutputCh.\tdev. name")

        for k in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(k)
            self.devices.append(dev)
            device_name = dev["name"]
            device_index = dev["index"]
            maxInputChannels = int(dev["maxInputChannels"])
            maxOutputChannels = int(dev["maxOutputChannels"])

            if type(device_name) is bytes:
                device_name = device_name.decode("cp932")  # for windows

            print(
                f"{device_index}\t{maxInputChannels}\t{maxOutputChannels}\t{device_name}"
            )

            # Prefer explicit index when provided.
            if input_device_index is not None and device_index == input_device_index:
                self.input_device_index = dev["index"]
                self.input_device_name = device_name
                self.RATE = int(dev["defaultSampleRate"])
                self.CHANNELS = dev["maxInputChannels"]
            # Fall back to keyword matching when index is not provided.
            elif (
                input_device_index is None
                and input_device_keyword in device_name
                and maxInputChannels == self.maxInputChannels
            ):
                self.input_device_index = dev["index"]
                self.input_device_name = device_name
                self.RATE = int(dev["defaultSampleRate"])
                self.CHANNELS = dev["maxInputChannels"]

        if self.input_device_index is not None:
            print("=========================================================")
            print(f"Input device:  {self.input_device_name} is OK.")
            print(f"\tRATE:      {self.RATE}")
            print(f"\tCHANNELS:  {self.CHANNELS}")
            print(f"\tCHUNK:     {self.CHUNK}")
            print("=========================================================")
        else:
            print("\nWarning: Input device is not exist\n")
            # Fallback to default input if available
            try:
                default_index = self.p.get_default_input_device_info()["index"]
                dev = self.p.get_device_info_by_index(default_index)
                self.input_device_index = dev["index"]
                self.input_device_name = dev["name"]
                self.RATE = int(dev["defaultSampleRate"])
                self.CHANNELS = int(dev["maxInputChannels"]) or self.CHANNELS
                print(f"Fallback to default input device: {self.input_device_name}")
            except Exception:
                pass

        # Open a read-only (input=True) stream; frames are pulled in `run()`.
        self.stream = self.p.open(
            format=self.format,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.input_device_index,
        )

        return None

    def run(self, callback_sigproc: Callable[[np.ndarray], None]) -> None:
        """Continuously read frames and feed them to a processing callback.

        The callback receives a 2D NumPy array of shape ``[channels, chunk]``,
        deinterleaved from the raw interleaved device buffer. The loop runs
        until the stream becomes inactive or an exception escapes.

        Args:
            callback_sigproc: A function that accepts a float/int NumPy array
                of shape ``[channels, chunk]`` and performs downstream work
                (e.g., feature extraction or visualization).
        """
        try:
            while self.stream.is_active():
                input_buff = self.stream.read(self.CHUNK, exception_on_overflow=False)
                # Convert the raw buffer to a NumPy array of the correct dtype.
                data = np.frombuffer(input_buff, dtype=self.dtype)
                # Check if the buffer contains the expected number of samples.
                expected = self.CHUNK * self.CHANNELS
                if data.size != expected:
                    # Skip incomplete frame to preserve downstream shape invariants.
                    continue
                # Deinterleave: [L, R, L, R, ...] → [[L...],[R...]] with shape [C, N].
                sig = np.reshape(data, (self.CHUNK, self.CHANNELS)).T
                callback_sigproc(sig)
        finally:
            self.__terminate()

    def __terminate(self) -> None:
        """Best-effort shutdown of stream and host without raising."""
        try:
            if self.stream.is_active():
                self.stream.stop_stream()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass

    # Public close for context-manager style
    def close(self) -> None:
        """Close the stream and terminate the PyAudio host."""
        self.__terminate()

    def __enter__(self) -> "AudioInputStream":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager."""
        self.__terminate()


def test_callback_sigproc(sig):
    """Test callback function for the audio stream."""
    print(sig.shape)


if __name__ == "__main__":
    ais = AudioInputStream()
    print(ais.get_params())
    ais.run(test_callback_sigproc)
