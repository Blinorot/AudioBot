import nemo.collections.asr as nemo_asr
import torch
from pyctcdecode import build_ctcdecoder
import kenlm
from pathlib import Path


def init_asr_model(device):
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name="QuartzNet15x5Base-En",
            strict=False,
            map_location=device
        )
    
    asr_model.eval()

    root_path = Path(__file__).absolute().resolve().parent.parent
    data_path = root_path / "data"
    lm_path = str(data_path / "lowercase_3-gram.pruned.1e-7.arpa")
    vocab_path = str(data_path / "librispeech-vocab.txt")

    with open(vocab_path) as f:
        unigram_list = [t.lower() for t in f.read().strip().split("\n")]

    ctc_decoder = build_ctcdecoder(
        asr_model.decoder.vocabulary,
        lm_path,
        unigram_list
    )
    
    return asr_model, ctc_decoder


def run_asr_model(asr_model, ctc_decoder, audio):
    output = asr_model(input_signal=audio, 
                       input_signal_length=torch.tensor([audio.shape[1]]))
    logprobs = output[0].numpy()

    text_output = ctc_decoder.decode(logprobs[0])
    
    print(text_output)
    return text_output