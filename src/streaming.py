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
KWS_THRESHOLD = 0.7  # threshold for assistant activation
SILENCE_PROPORTION_IN_CHUNK = 0.5  # VAD hyperparameter
IS_SILENCE = 1
IS_SPEECH = 0


def vad_checking(chunk, sample_rate):
    """
    Simple wrapper over torchaudio Voice Activity Detector (VAD).

    Used as a simple approach to understand that user finished the query.
    Returns "IS_SILENCE" if the proportion of voice activity in the chunk
    is less than SILENCE_PROPORTION_IN_CHUNK.

    We use proportion because the algorithm for VAD is simple and may
    consider some noise as speech.

    We will say that the user stopped talking if there are vad_limit
    silent successive chunks.

    Args:
        chunk (torch.Tensor): audio chunk.
        sample_rate (int): audio sample rate.
    Returns:
        IS_SILENCE | IS_SPEECH: the chunk type.
    """
    vad_chunk = vad(chunk, sample_rate=sample_rate)
    # some rule to identify if the chunk contains speech
    if vad_chunk.shape[1] <= chunk.shape[1] * SILENCE_PROPORTION_IN_CHUNK:
        return IS_SILENCE
    else:
        return IS_SPEECH


def get_kws():
    """
    Get pretrained Keyword-Spotting System (KWS). Used to activate assistant.
    For the keyword, the word *sheila* is used.

    Returns:
        kws: pretrained KWS system.
    """
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
    """
    A separate process to record user's query.
    Initializes microphone recording, gets user's query,
    sends it to the main process for processing and output generation.

    KWS and VAD are done in this process because they are fast.
    Ideally, a separate process should be used for all processing features.

    Communication with the main process is done via 2 queues.

    Args:
        init_queue: communication queue. Used to control the start
            and end of the recording process.
        queue: communication queue. Used to send user's query chunks
            to the main process.
        source: parameter for StreamReader (microphone id).
        format: parameter for StreamReader (microphone format, see README).
        sample_rate: audio sample rate for the microphone recording.
        vad_limit: number of successive silent chunks required to
            stop the recording.
    """
    kws = get_kws()
    while init_queue.get() == SIG_START:  # main process sent the signal to start
        streamer = StreamReader(src=source, format=format)
        streamer.add_basic_audio_stream(
            frames_per_chunk=chunk_size, sample_rate=sample_rate, stream_index=0
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
                    # the assistant is activated
                    print(f"Keyword Detected. Probability: {keyword_proba}")
                    query_started = True
                else:
                    continue

            # send chunk to the main process for processing
            # and output generation
            queue.put(chunk_data)

            # check stop criteria: vad_limit successive silent chunks
            vad_check = vad_checking(chunk_data, sample_rate=sample_rate)
            if vad_check:
                vad_counts += 1
            else:
                vad_counts = 0
                user_talked = True
            if user_talked and vad_counts >= vad_limit:
                break

        queue.put(SIG_STOP)  # stop criteria for the main process loop
        print("End of the user input")
        streamer.remove_stream(0)  # remove stream with id=0


def init_stream(all_models, history, config):
    """
    Main Process.

    Initializes all multiprocessing managers and handlers.
    Creates a separate process for recording and communicates
    with it to get user's query. Then, processes the query and
    play the audio response.

    A separate process could be used for audio playback. For simplicity,
    we use the main process for playback too. The latency issue is mitigated
    because the process is still able to generate new chunks, while the old
    ones are playing.

    Use Keyboard Interrupt (Ctrl+C) to stop the program.

    Args:
        all_models (list): list of all returned models.
        history (list[dict]): chat history.
        config: Hydra config to control processing.
    """

    # initialize multiprocessing stuff
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    chunk_queue = manager.Queue(maxsize=0)
    stream_init_queue = manager.Queue(maxsize=0)

    # initialize recording process
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

    # initialize writer for audio output playback
    stream_writer = StreamWriter(
        dst=config.stream_writer.dst, format=config.stream_writer.format
    )
    stream_writer.add_audio_stream(
        sample_rate=config.stream_writer.sample_rate, num_channels=1
    )
    stream_writer.open()

    # start the recording process
    streaming_process.start()
    stream_init_queue.put(SIG_START)  # sent signal to start
    while True:
        try:
            while True:
                chunk = chunk_queue.get()
                if isinstance(chunk, int):
                    # stop criteria achieved
                    # chunk == SIG_STOP
                    break
                # concatenate successive query chunks
                user_chunks.append(chunk)

            # send it to ASR, LLM and TTS
            # get output audio generator
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

                # write the chunks to the StreamWriter for a playback
                for i in range(0, num_frames, config.stream_writer.chunk_size):
                    stream_writer.write_audio_chunk(
                        0, user_audio_output[i : i + config.stream_writer.chunk_size]
                    )

            audio_time = total_num_frames / config.stream_writer.sample_rate
            end_time = time.perf_counter()

            # to avoid reading from the microphone before the output is finished
            # we calculate the audio output time and the time we used to
            # process the audio. We stop the process for the difference time,
            # so the audio output finished before we move on.
            diff_time = audio_time - (end_time - start_time)
            if diff_time > 0:
                print(f"Audio time: {audio_time}. Have to sleep for {diff_time}.")
                # 1 extra second to be sure
                time.sleep(1 + diff_time)  # to avoid ASR reading an output

            # init next user input
            user_chunks = []
            stream_init_queue.put(SIG_START)

        except KeyboardInterrupt:
            # stop the program
            break
        except Exception as exc:
            raise exc

    # send signal to exit
    stream_init_queue.put(SIG_STOP)
    streaming_process.join()
    stream_writer.close()


if __name__ == "__main__":
    init_stream()
