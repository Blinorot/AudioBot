import logging
import multiprocessing as mp
import time

import torch
from torchaudio.functional import vad
from torchaudio.io import StreamReader, StreamWriter

from src.handlers import full_handler

logger = logging.getLogger(__file__)


def vad_checking(chunk, sample_rate):
    vad_chunk = vad(chunk, sample_rate=sample_rate)
    if vad_chunk.shape[1] <= chunk.shape[1] * 0.5:
        return 1
    else:
        return 0


def audio_stream(
    init_queue: mp.Queue, queue: mp.Queue, source, format, chunk_size, sample_rate
):
    streamer = StreamReader(src=source, format=format)
    while init_queue.get() == 1:  # main process sent the signal to start
        streamer.add_basic_audio_stream(
            frames_per_chunk=chunk_size, sample_rate=sample_rate
        )
        stream_iterator = streamer.stream(timeout=-1, backoff=1.0)

        vad_counts = 0
        user_talked = 0
        vad_limit = 3

        print("Start audio streaming")
        while True:
            (chunk_,) = next(stream_iterator)
            chunk_data = chunk_.data
            chunk_data = chunk_data.sum(dim=-1)  # to mono
            chunk_data = chunk_data.view(1, -1)
            queue.put(chunk_data)
            vad_check = vad_checking(chunk_data, sample_rate=sample_rate)
            if vad_check:
                vad_counts += 1
            else:
                vad_counts = 0
                user_talked = 1
            if user_talked and vad_counts >= vad_limit:
                break
            # if vad_counts > 5:
            #    break
        queue.put(0)  # stop criteria for main process loop
        print("End of the user input")
        streamer.remove_stream(0)


def init_stream(
    all_models,
    history,
    device,
    source="hw:0",
    format="alsa",
    chunk_size=16000,
    sample_rate=16000,
):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    chunk_queue = manager.Queue(maxsize=0)
    stream_init_queue = manager.Queue(maxsize=0)
    streaming_process = ctx.Process(
        target=audio_stream,
        args=(stream_init_queue, chunk_queue, source, format, chunk_size, sample_rate),
    )

    user_chunks = []

    stream_writer = StreamWriter(dst="default", format="alsa")
    stream_writer.add_audio_stream(sample_rate=22050, num_channels=1)
    stream_writer.open()

    streaming_process.start()
    stream_init_queue.put(1)  # sent signal to start
    while True:
        try:
            while True:
                chunk = chunk_queue.get()
                if isinstance(chunk, int):  # stop criteria achieved
                    break
                user_chunks.append(chunk)

            # send it to ASR, LLM and TTS
            user_full = torch.cat(user_chunks, dim=-1)
            user_audio_output, history = full_handler(
                all_models, history, device, user_full
            )

            user_audio_output = user_audio_output[0].unsqueeze(-1)
            num_frames = user_audio_output.shape[0]
            for i in range(0, num_frames, 256):
                stream_writer.write_audio_chunk(0, user_audio_output[i : i + 256])
            time.sleep(1 + num_frames / 22050)  # to avoid ASR reading an output

            # init next user input
            user_chunks = []
            stream_init_queue.put(1)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    # send signal to exit
    stream_init_queue.put(0)
    streaming_process.join()
    stream_writer.close()


if __name__ == "__main__":
    init_stream()
