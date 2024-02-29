import torch
from pathlib import Path
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan


def init_tts_model(device):
    tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
    spec_generator = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
    vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
   
    return spec_generator, tokenizer, vocoder


def run_tts_model(text, spec_generator, tokenizer, vocoder):
    # Generate audio
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    output_dict = spec_generator(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]

    audio = vocoder(spectrogram)
    return audio