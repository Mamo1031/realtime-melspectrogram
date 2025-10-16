"""Audio input streaming utility (prints a compact device table on startup)."""

import pyaudio
import numpy as np
from typing import Callable, Dict, List, Optional


class AudioInputStream:
    """Capture audio frames with simple device selection and callback delivery."""

    def __init__(
        self,
        format: int = pyaudio.paFloat32,
        input_device_keyword: str = "VoiceMeeter Output",
        CHUNK: int = 1024,
        maxInputChannels: int = 2,
        input_device_index: Optional[int] = None,
    ) -> None:
        """Open an input device and prepare fixed-size frame capture."""
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
        """Return ``{"RATE","CHUNK","CHANNELS"}`` mapping."""
        params_dict = {
            "RATE": self.RATE,
            "CHUNK": self.CHUNK,
            "CHANNELS": self.CHANNELS,
        }
        return params_dict

    @staticmethod
    def list_input_devices() -> List[Dict]:
        """Return raw device info dicts visible to PyAudio."""
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
        """Pick device by index/keyword (fallback to default) and open stream."""
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
        """Read frames in a loop and pass ``[channels, chunk]`` arrays to callback."""
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
        """Best-effort shutdown without raising."""
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
        """Close resources."""
        self.__terminate()

    def __enter__(self) -> "AudioInputStream":
        """Context enter."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context exit."""
        self.__terminate()


def test_callback_sigproc(sig):
    print(sig.shape)


if __name__ == "__main__":
    ais = AudioInputStream()
    print(ais.get_params())
    ais.run(test_callback_sigproc)
