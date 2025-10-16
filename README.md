# realtime-melspectrogram
A project for real-time audio capture and mel-spectrogram visualization. This project uses PyAudio for microphone input.

Before running the project, make sure PortAudio (the native dependency for PyAudio) is installed on your system. Otherwise, pyaudio installation will fail.


## Requirements

* Python 3.12
* uv (modern Python package manager)
* PortAudio (system dependency for PyAudio)


## System Setup for Ubuntu
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```


## Python Environment Setup

```bash
# Create a virtual environment
uv venv

# Sync dependencies (refer to pyproject.toml for installation)
uv sync
```
