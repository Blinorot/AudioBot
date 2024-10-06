import uuid

import torch
import torchaudio

from src.asr import init_asr_model, run_asr_model
from src.llm import init_llm, run_llm
from src.tts import init_tts_model, run_tts_model


def init_all_models(device="cpu"):
    asr_model, processor = init_asr_model(device)
    spec_generator, spec_tokenizer, vocoder = init_tts_model(device)
    client, history = init_llm()

    all_models = [
        asr_model,
        processor,
        spec_generator,
        spec_tokenizer,
        vocoder,
        client,
    ]
    return all_models, history


def user_audio_input_handler(user_audio_input, asr_model, processor):
    return run_asr_model(asr_model, processor, user_audio_input)


def user_text_input_handler(user_text_input, history, client):
    return run_llm(user_text_input, history, client)


def user_output_handler(user_text_output, spec_generator, spec_tokenizer, vocoder):
    return run_tts_model(user_text_output, spec_generator, spec_tokenizer, vocoder)


@torch.inference_mode()
def full_handler(all_models, history, device, user_audio_input):
    (
        asr_model,
        ctc_decoder,
        spec_generator,
        spec_tokenizer,
        vocoder,
        client,
    ) = all_models

    user_text_input = user_audio_input_handler(user_audio_input, asr_model, ctc_decoder)
    user_text_output, history = user_text_input_handler(
        user_text_input, history, client
    )
    user_audio_output = user_output_handler(
        user_text_output, spec_generator, spec_tokenizer, vocoder
    )
    # torchaudio.save(f"data/audio_{uuid.uuid4()}.wav", user_audio_output, sample_rate=22050)

    print("Text Input:", user_text_input)
    print("Audio Output Shape:", user_audio_output.shape)

    return user_audio_output, history
