import torch
import torchaudio
import uuid

from src.asr import init_asr_model, run_asr_model
from src.tts import init_tts_model, run_tts_model
from src.llm import init_llm, run_llm


def init_all_models(device="cpu"):
    asr_model, ctc_decoder = init_asr_model(device)
    spec_generator, vocoder = init_tts_model(device)
    #model, tokenizer, streamer = init_llm()
    #spec_generator=None
    #vocoder=None
    model=None
    tokenizer=None
    streamer=None

    past_key_values=None
    sequence=None
    seq_len=0

    all_models = [asr_model, ctc_decoder, spec_generator,
                  vocoder, model, tokenizer, streamer]
    metadata = [past_key_values, sequence, seq_len]
    return all_models, metadata


def user_audio_input_handler(user_audio_input, asr_model, ctc_decoder):
    return run_asr_model(asr_model, ctc_decoder, user_audio_input)


def user_text_input_handler(user_text_input, model, tokenizer, 
                            streamer, device, past_key_values=None,
                            sequence=None, seq_len=0):
    return run_llm(model, tokenizer, streamer,
                   text=user_text_input, device=device,
                   past_key_values=past_key_values,
                   sequence=sequence, seq_len=seq_len)


def user_output_handler(user_text_output, spec_generator, vocoder):
    return run_tts_model(user_text_output, spec_generator, vocoder)


@torch.inference_mode()
def full_handler(all_models, metadata, device, user_audio_input):
    asr_model, ctc_decoder, spec_generator, vocoder,\
                        model, tokenizer, streamer = all_models
    past_key_values, sequence, seq_len = metadata

    user_text_input = user_audio_input_handler(user_audio_input, asr_model,
                                               ctc_decoder)
    # user_text_output = user_text_input_handler(user_text_input, model, 
    #                                            tokenizer, streamer,
    #                                            device, past_key_values,
    #                                            sequence, seq_len)
    user_text_output = user_text_input
    user_audio_output = user_output_handler(user_text_output, spec_generator, vocoder)
    torchaudio.save(f"data/audio_{uuid.uuid4()}.wav", user_audio_output, sample_rate=22050)

    print(user_text_input)
    print(user_audio_output.shape)
    
    metadata = [past_key_values, sequence, seq_len]
    return metadata