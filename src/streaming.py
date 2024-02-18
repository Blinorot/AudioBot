import torch
from torchaudio.io import StreamReader
import torchaudio
from torchaudio.functional import vad

import multiprocessing as mp
import logging


from src.handlers import init_all_models, full_handler


logger = logging.getLogger(__file__)


def vad_checking(chunk, sample_rate):
    vad_chunk = vad(chunk, sample_rate=sample_rate)
    if vad_chunk.shape[1] <= chunk.shape[1] * 0.5:
        return 1
    else:
        return 0


def audio_stream(queue: mp.Queue, source, format, chunk_size, sample_rate):
    streamer = StreamReader(src=source, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=chunk_size, sample_rate=sample_rate)
    stream_iterator = streamer.stream(timeout=-1, backoff=1.0)

    print("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        chunk_data = chunk_.data
        queue.put(chunk_data)
        


def init_stream(source="hw:0", format="alsa", chunk_size=16000, sample_rate=16000, device="cpu"):
    all_models, metadata = init_all_models(device)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    chunk_queue = manager.Queue(maxsize=0)
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue, source,
                                                               format, chunk_size, sample_rate))

    vad_counts = 0
    limit = 3

    user_chunks = []

    streaming_process.start()
    while True:
        try:
            chunk = chunk_queue.get()
            chunk = chunk.sum(dim=-1) # to mono
            chunk = chunk.view(1, -1)

            vad_check = vad_checking(chunk, sample_rate=sample_rate)
            if vad_check:
                vad_counts += 1
            else:
                vad_counts = 0

            user_chunks.append(chunk)

            if vad_counts >= limit:
                print("End of the user input")
                # send it to ASR
                user_full = torch.cat(user_chunks, dim=-1)
                metadata = full_handler(all_models, metadata, device, user_full)
                # reset for next
                user_chunks = []

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()


if __name__ == "__main__":
    init_stream()
