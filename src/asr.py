import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def init_asr_model(config):
    """
    Initialize speech recognition system.
    In this example, we use general class. The exact model can be provided
    via config.asr.model_id (id from HuggingFace).

    Args:
        config: Hydra config to control ASR and device.
    Returns:
        asr_pipeline: HF pipeline to convert speech into text.
    """
    torch_dtype = torch.float32 if config.device == "cpu" else torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(config.device)

    processor = AutoProcessor.from_pretrained(config.asr.model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=config.device,
    )
    return asr_pipeline


def run_asr_model(asr_pipeline, audio):
    """
    Get transcription for a speech input.

    Args:
        asr_pipeline: HF pipeline to get text from audio input.
        audio (torch.Tensor): input audio.
    Returns:
        text_output (str): text transcription of user's query.
    """
    # pipeline expects numpy array of shape (T)
    # so we convert and take 0-th channel
    text_output = asr_pipeline(audio.numpy()[0])["text"]

    return text_output
