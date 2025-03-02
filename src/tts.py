from transformers import (
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerTokenizer,
)


def init_tts_model(config):
    """
    Initialize TTS model.
    For this example, we use specific classes for FastSpeech2 and HiFiGAN
    instead of a general class.

    Args:
        config: Hydra config to get device.
    Returns:
        tokenizer: tokenizer to encoder text input.
        spec_generator: FastSpeech2 to generate a mel spectrogram from text.
        vocoder: HiFiGAN to generate a waveform from a mel spectrogram.
    """
    tokenizer = FastSpeech2ConformerTokenizer.from_pretrained(
        "espnet/fastspeech2_conformer",
    )
    spec_generator = FastSpeech2ConformerModel.from_pretrained(
        "espnet/fastspeech2_conformer",
    )
    vocoder = FastSpeech2ConformerHifiGan.from_pretrained(
        "espnet/fastspeech2_conformer_hifigan",
    )

    spec_generator.to(config.device)
    vocoder.to(config.device)

    return tokenizer, spec_generator, vocoder


def run_tts_model(text, tokenizer, spec_generator, vocoder, config):
    """
    Generate audio from text input.

    To reduce latency and play audio, while still processing successive
    chunks, we split text into small pieces. This ensures that TTS is
    fast enough and the user is not waiting for too long for the response.
    In this example, we simply partition text into chunks of size
    config.tts.max_words_per_query.

    Args:
        text (str): text input (LLM response).
        tokenizer: tokenizer to encoder text input.
        spec_generator: FastSpeech2 to generate a mel spectrogram from text.
        vocoder: HiFiGAN to generate a waveform from a mel spectrogram.
        config: Hydra config for configuration of splitting algorithm.
    Returns:
        audio: generator, containing yielded waveforms for each partition
            of the input text.
    """
    max_words_per_query = config.tts.max_words_per_query

    text_partitions = text.split()
    returned_partitions = 0
    number_of_partitions = len(text_partitions) // max_words_per_query
    for i in range(number_of_partitions + 1):
        # for a case when len(text_partitions) % max_words_per_query == 0
        if i * max_words_per_query == len(text_partitions):
            break
        query = text_partitions[i * max_words_per_query : (i + 1) * max_words_per_query]
        returned_partitions += len(query)

        query = " ".join(query)

        # some logging
        print(f"Response ({returned_partitions}/{len(text_partitions)}): {query}")

        # Generate audio
        # text -> encoded_text -> mel spec -> waveform
        inputs = tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"]
        output_dict = spec_generator(input_ids.to(config.device), return_dict=True)
        spectrogram = output_dict["spectrogram"]

        audio = vocoder(spectrogram).detach().cpu()
        yield audio
