import nemo.collections.tts as nemo_tts


def init_tts_model(device):
    # Load mel spectrogram generator
    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch",
                                                                    map_location=device)
    # Load vocoder
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_en_hifigan",
                                                           map_location=device)
    
    spec_generator.eval()
    vocoder.eval()
   
    return spec_generator, vocoder


def run_tts_model(text, spec_generator, vocoder):
    # Generate audio
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    # Save the audio to disk in a file called speech.wav
    #sf.write("speech.wav", audio.to('cpu').numpy(), 22050)
    return audio