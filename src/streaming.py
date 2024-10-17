import multiprocessing as mp
import time
from pathlib import Path

import torch
import wget
from torchaudio.functional import vad
from torchaudio.io import StreamReader, StreamWriter

from src.handlers import KWS_URL, full_handler

SIG_START = 1
SIG_STOP = 0
KWS_THRESHOLD = 0.7
SILENCE_PROPORTION_IN_CHUNK = 0.5
IS_SILENCE = 1
IS_SPEECH = 0


def vad_checking(chunk, sample_rate):
    vad_chunk = vad(chunk, sample_rate=sample_rate)
    # some rule to identiy
    if vad_chunk.shape[1] <= chunk.shape[1] * SILENCE_PROPORTION_IN_CHUNK:
        return IS_SILENCE
    else:
        return IS_SPEECH


def get_kws():
    data_path = Path(__file__).absolute().resolve().parent.parent / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    kws_path = data_path / "kws.pth"
    if not kws_path.exists():
        wget.download(KWS_URL, str(kws_path))
    kws = torch.jit.load(kws_path, map_location="cpu")
    return kws


def audio_stream(
    init_queue: mp.Queue,
    queue: mp.Queue,
    source,
    format,
    chunk_size,
    sample_rate,
    vad_limit,
):
    kws = get_kws()
    streamer = StreamReader(src=source, format=format)
    while init_queue.get() == SIG_START:  # main process sent the signal to start
        streamer.add_basic_audio_stream(
            frames_per_chunk=chunk_size, sample_rate=sample_rate
        )
        stream_iterator = streamer.stream(timeout=-1, backoff=1.0)

        vad_counts = 0
        user_talked = False

        query_started = False

        print("Start audio streaming")
        while True:
            (chunk_,) = next(stream_iterator)
            chunk_data = chunk_.data
            chunk_data = chunk_data.sum(dim=-1)  # to mono
            chunk_data = chunk_data.view(1, -1)

            if not query_started:
                # check that the query is started
                with torch.inference_mode():
                    keyword_proba = kws(chunk_data)
                if keyword_proba > KWS_THRESHOLD:
                    print(f"Keyword Detected: {keyword_proba}")
                    query_started = True
                else:
                    continue

            queue.put(chunk_data)
            vad_check = vad_checking(chunk_data, sample_rate=sample_rate)
            if vad_check:
                vad_counts += 1
            else:
                vad_counts = 0
                user_talked = True
            if user_talked and vad_counts >= vad_limit:
                break

        queue.put(SIG_STOP)  # stop criteria for main process loop
        print("End of the user input")
        streamer.remove_stream(0)  # remove stream with id=0


def init_stream(all_models, history, config):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    chunk_queue = manager.Queue(maxsize=0)
    stream_init_queue = manager.Queue(maxsize=0)
    streaming_process = ctx.Process(
        target=audio_stream,
        args=(
            stream_init_queue,
            chunk_queue,
            config.stream_reader.source,
            config.stream_reader.format,
            config.stream_reader.chunk_size,
            config.stream_reader.sample_rate,
            config.stream_reader.vad_limit,
        ),
    )

    user_chunks = []

    stream_writer = StreamWriter(
        dst=config.stream_writer.dst, format=config.stream_writer.format
    )
    stream_writer.add_audio_stream(
        sample_rate=config.stream_writer.sample_rate, num_channels=1
    )
    stream_writer.open()

    streaming_process.start()
    stream_init_queue.put(SIG_START)  # sent signal to start
    while True:
        try:
            while True:
                chunk = chunk_queue.get()
                if isinstance(chunk, int):  # stop criteria achieved
                    break
                user_chunks.append(chunk)

            # send it to ASR, LLM and TTS
            user_full = torch.cat(user_chunks, dim=-1)
            user_audio_output_generator, history = full_handler(
                all_models, history, user_full, config
            )

            start_time = time.perf_counter()
            total_num_frames = 0
            for user_audio_output in user_audio_output_generator:
                user_audio_output = user_audio_output[0].unsqueeze(-1)
                num_frames = user_audio_output.shape[0]
                total_num_frames += num_frames
                for i in range(0, num_frames, config.stream_writer.chunk_size):
                    stream_writer.write_audio_chunk(
                        0, user_audio_output[i : i + config.stream_writer.chunk_size]
                    )
            audio_time = total_num_frames / config.stream_writer.sample_rate
            end_time = time.perf_counter()

            diff_time = audio_time - (end_time - start_time)
            if diff_time > 0:
                print(audio_time, diff_time)
                time.sleep(1 + diff_time)  # to avoid ASR reading an output

            # init next user input
            user_chunks = []
            stream_init_queue.put(SIG_START)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    # send signal to exit
    stream_init_queue.put(SIG_STOP)
    streaming_process.join()
    stream_writer.close()


if __name__ == "__main__":
    init_stream()
