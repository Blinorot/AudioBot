# AudioBot

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an implementation of an intelligent voice assistant. The solution is based on the combination of Automatic Speech Recognition (ASR), Text To Speech (TTS), and Large Language Models (LLM) systems.

The assistant is activated using a Keyword-Spotting system (KWS) with `sheila` as a target word. Then, the user says the query and an ASR model converts speech query into text. The text query is given as input to an LLM, and its response is converted back to audio using a TTS system. After the audio playback is finished, the user can continue the dialogue. The LLM preserves the history of the chat.

The version with default choice of models works fast even on CPU! For better transcription quality, consider using a different ASR model from HuggingFace (e.g. `openai/whisper-large-v2` with a GPU instead of CPU to make it work fast enough).

See the [LauzHack Workshop](https://youtu.be/rK4I-F8Y6pw) with the discussion on how to create intelligent voice assistants and this repository (also see [Slides](https://docs.google.com/presentation/d/1r0vdgrl7nbSjNQcszLk_A12jlArDUiuESHsYh40yaDo/edit?usp=sharing)).

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
   source project_env/bin/activate
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

To record and play sound, you need to define your hardware settings. See more in the [PyTorch documentation](https://pytorch.org/audio/2.2.0/generated/torio.io.StreamingMediaDecoder.html#torio.io.StreamingMediaDecoder) (information about `ffmpeg` specifically) and [this tutorial](https://pytorch.org/audio/2.4.0/tutorials/streamreader_advanced_tutorial.html). Usually, the format is `alsa` for Linux systems and `avfoundation` for Mac systems. For the reader source and writer dst, the `default` option usually works (so it might be enough to change the format only in your case).

When the hardware is known, you can start AI AudioBot using this command:

```bash
python3 run.py stream_reader.source=YOUR_MICROPHONE \
    stream_reader.format=YOUR_FORMAT \
    stream_writer.dst=YOUR_LOUDSPEAKER \
    stream_writer.format=YOUR_FORMAT
```

You can also change other parameters via Hydra options. See `src/configs/audio_bot.yaml`. For example, you can change the maximum number of output tokens and LLM model:

```bash
python3 run.py llm.model_id="mixtral-8x7b-32768" llm.max_tokens=256
```

Use `Keyboard Interrupt` (`ctrl+C`) to stop the assistant.

## Credits

[HuggingFace](https://huggingface.co/) was used for [ASR](https://huggingface.co/spaces/openai/whisper) and TTS models ([Spectrogram Generator](https://huggingface.co/espnet/fastspeech2_conformer) and [Vocoder](https://huggingface.co/espnet/fastspeech2_conformer_hifigan)). [Groq API](https://groq.com/) with [llama-3-8b-8192](https://ai.meta.com/blog/meta-llama-3/) model was used for LLM. The KWS model is taken from the 2022 version of the [HSE DLA Course](https://github.com/markovka17/dla).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
