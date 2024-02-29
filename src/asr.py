import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def init_asr_model(device):
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return model, processor


def run_asr_model(asr_model, processor, audio):
    input_features = processor(
        audio[0], sampling_rate=16000, return_tensors="pt"
    ).input_features

    predicted_ids = asr_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    text_output = transcription[0]
    
    print("ASR output:", text_output)
    return text_output