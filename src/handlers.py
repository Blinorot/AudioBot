from pathlib import Path

import torch
import wget

from src.asr import init_asr_model, run_asr_model
from src.llm import init_llm, run_llm
from src.tts import init_tts_model, run_tts_model

KWS_URL = "https://github.com/markovka17/dla/raw/refs/heads/2022/hw2_kws/kws.pth"


def init_all_models(config):
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
    return run_asr_model(asr_pipeline, user_audio_input)


def user_text_input_handler(user_text_input, history, client, config):
    return run_llm(user_text_input, history, client, config)


def user_output_handler(user_text_output, tokenizer, spec_generator, vocoder, config):
    return run_tts_model(user_text_output, tokenizer, spec_generator, vocoder, config)


@torch.inference_mode()
def full_handler(all_models, history, user_audio_input, config):
    (
        asr_pipeline,
        tokenizer,
        spec_generator,
        vocoder,
        client,
    ) = all_models

    user_text_input = user_audio_input_handler(user_audio_input, asr_pipeline)
    print("Text Input:", user_text_input)

    user_text_output, history = user_text_input_handler(
        user_text_input, history, client, config
    )

    user_audio_output_generator = user_output_handler(
        user_text_output, tokenizer, spec_generator, vocoder, config
    )
    return user_audio_output_generator, history
