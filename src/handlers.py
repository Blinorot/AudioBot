import uuid

import torch
import torchaudio

from src.asr import init_asr_model, run_asr_model
from src.llm import init_llm, run_llm
from src.tts import init_tts_model, run_tts_model


def init_all_models(config):
    asr_pipeline = init_asr_model(config)
    tokenizer, spec_generator, vocoder, max_words_per_query = init_tts_model(config)
    client, history = init_llm()

    all_models = [
        asr_pipeline,
        tokenizer,
        spec_generator,
        vocoder,
        max_words_per_query,
        client,
    ]
    return all_models, history


def user_audio_input_handler(user_audio_input, asr_pipeline):
    return run_asr_model(asr_pipeline, user_audio_input)


def user_text_input_handler(user_text_input, history, client):
    return run_llm(user_text_input, history, client)


def user_output_handler(
    user_text_output, tokenizer, spec_generator, vocoder, max_words_per_query
):
    return run_tts_model(
        user_text_output, tokenizer, spec_generator, vocoder, max_words_per_query
    )


@torch.inference_mode()
def full_handler(all_models, history, user_audio_input):
    (
        asr_pipeline,
        tokenizer,
        spec_generator,
        vocoder,
        max_words_per_query,
        client,
    ) = all_models

    user_text_input = user_audio_input_handler(user_audio_input, asr_pipeline)
    print("Text Input:", user_text_input)

    user_text_output, history = user_text_input_handler(
        user_text_input, history, client
    )

    user_audio_output_generator = user_output_handler(
        user_text_output, tokenizer, spec_generator, vocoder, max_words_per_query
    )
    return user_audio_output_generator, history
