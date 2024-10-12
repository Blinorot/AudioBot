# AudioBot

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contain an implementation of an intelligent voice assistant. The solution is based on the combination of Automatic Speech Recognition, Text To Speech, and LLM models.

## Installation

To install the assistant, follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install `pre-commit`:

   ```bash
   pre-commit install
   ```

3. Create an API key in [Groq](https://groq.com/). Create a new file named `.env` in the root directory and copy-paste your API key into it.

## How To Use

To record and play sound, you need to define your hardware settings. See more in the [PyTorch documentation](https://pytorch.org/audio/2.2.0/generated/torio.io.StreamingMediaDecoder.html#torio.io.StreamingMediaDecoder) (information about `ffmpeg` specifically) and [this tutorial](https://pytorch.org/audio/2.4.0/tutorials/streamreader_advanced_tutorial.html). Usually, the format is `alsa` for linux systems and `avfoundation` for mac systems.

When the hardware is known, you can start AI AudioBot using this command:

```bash
python3 run.py stream_reader.source=YOUR_MICROPHONE \
    stream_reader.format=YOUR_FORMAT \
    stream_writer.format=YOUR_FORMAT
```

## Credits

[HuggingFace](https://huggingface.co/) was used for [ASR](https://huggingface.co/spaces/openai/whisper) and TTS models ([Spectrogram Generator](https://huggingface.co/espnet/fastspeech2_conformer) and [Vocoder](https://huggingface.co/espnet/fastspeech2_conformer_hifigan)). [Groq API](https://groq.com/) with [llama-3-8b-8192](https://ai.meta.com/blog/meta-llama-3/) model was used for LLM.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
