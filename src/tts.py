from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerTokenizer,
)


def init_tts_model(config):
    tokenizer = FastSpeech2ConformerTokenizer.from_pretrained(
        "espnet/fastspeech2_conformer"
    )
    spec_generator = FastSpeech2ConformerModel.from_pretrained(
        "espnet/fastspeech2_conformer"
    )
    vocoder = FastSpeech2ConformerHifiGan.from_pretrained(
        "espnet/fastspeech2_conformer_hifigan"
    )

    max_words_per_query = config.tts.max_words_per_query

    return tokenizer, spec_generator, vocoder, max_words_per_query


def run_tts_model(text, tokenizer, spec_generator, vocoder, max_words_per_query):
    text_partitions = text.split()
    number_of_partitions = len(text_partitions) // max_words_per_query
    for i in range(number_of_partitions + 1):
        if i * max_words_per_query == len(text_partitions):
            break
        query = text_partitions[i * max_words_per_query : (i + 1) * max_words_per_query]
        query = " ".join(query)
        print(len(text_partitions), query)

        # Generate audio
        inputs = tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]
        output_dict = spec_generator(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"]

        audio = vocoder(spectrogram)
        yield audio
