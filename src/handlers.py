from pathlib import Path

import torch
import wget

from src.asr import init_asr_model, run_asr_model
from src.llm import init_llm, run_llm
from src.tts import init_tts_model, run_tts_model

# pre-trained KWS checkpoint
KWS_URL = "https://github.com/markovka17/dla/raw/refs/heads/2022/hw2_kws/kws.pth"


def init_all_models(config):
    """
    Initialize ASR, TTS, and LLM.

    Args:
        config: Hydra config to control initialization.
    Returns:
        all_models (list): list of all returned models.
        history (list[dict]): chat history.
    """
    asr_pipeline = init_asr_model(config)
    tokenizer, spec_generator, vocoder = init_tts_model(config)
    client, history = init_llm()

    all_models = [
        asr_pipeline,
        tokenizer,
        spec_generator,
        vocoder,
        client,
    ]
    return all_models, history


def user_audio_input_handler(user_audio_input, asr_pipeline):
    """
    Wrapper over asr model.
    Args:
        user_audio_input (torch.Tensor): audio input.
        asr_pipeline: HF pipeline to get text transcription.
    Returns:
        user_text_input (str): text transcription of user's query.
    """
    return run_asr_model(asr_pipeline, user_audio_input)


def user_text_input_handler(user_text_input, history, client, config):
    """
    Wrapper over LLM system.
    Args:
        user_text_input (str): input user query.
        history (list[dict]): chat history.
        client: Groq client for API calls.
        config: Hydra config with llm configuration in config.llm.
    Returns:
        predicted_text (str): LLM response.
        history (list[dict]): updated history.
    """
    return run_llm(user_text_input, history, client, config)


def user_output_handler(user_text_output, tokenizer, spec_generator, vocoder, config):
    """
    Wrapper over TTS system.
    Args:
        user_text_output (str): text input (LLM response).
        tokenizer: tokenizer to encoder text input.
        spec_generator: FastSpeech2 to generate a mel spectrogram from text.
        vocoder: HiFiGAN to generate a waveform from a mel spectrogram.
        config: Hydra config for configuration of splitting algorithm.
    Returns:
        user_audio_output_generator: generator, containing yielded waveforms for each partition
            of the input text.
    """
    return run_tts_model(user_text_output, tokenizer, spec_generator, vocoder, config)


@torch.inference_mode()
def full_handler(all_models, history, user_audio_input, config):
    """
    Full handler to process user's audio and return audio response.
    Args:
        all_models (list): list of all returned models.
        history (list[dict]): chat history.
        user_audio_input (torch.Tensor): audio input.
        config: Hydra config to control processing.
    Returns:
        user_audio_output_generator: generator, containing yielded waveforms for each partition
            of the input text.
        history (list[dict]): updated chat history.
    """
    (
        asr_pipeline,
        tokenizer,
        spec_generator,
        vocoder,
        client,
    ) = all_models

    # audio -> text
    user_text_input = user_audio_input_handler(user_audio_input, asr_pipeline)
    # logging
    print("Text Input:", user_text_input)

    # user query -> LLM response
    user_text_output, history = user_text_input_handler(
        user_text_input, history, client, config
    )

    # text response -> audio response
    user_audio_output_generator = user_output_handler(
        user_text_output, tokenizer, spec_generator, vocoder, config
    )
    return user_audio_output_generator, history
