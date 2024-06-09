import sounddevice as sd
import soundfile as sf
from pathlib import Path
import logging
import numpy as np
from pydub import playback, AudioSegment

logger = logging.getLogger(__name__)


_audio_dtype = np.float32


def record_audio(duration=5, samplerate=16000, file_output=Path('my_audio.wav')):
    try:
        logger.info("Recording...")

        # Allocate space for the recording
        myrecording = np.zeros(
            (int(duration * samplerate), 1), dtype=_audio_dtype)

        # Start recording
        sd.rec(out=myrecording, samplerate=samplerate,
               channels=1, dtype=_audio_dtype)

        # Wait until the recording is finished
        sd.wait()

        logger.info("Recording complete.")

        # Write the audio file
        sf.write(file_output, myrecording, samplerate)
        logger.info(f"Audio saved to {file_output}")
        return file_output
        # return myrecording, samplerate

    except Exception as e:
        logger.error(f"An error occurred during recording: {e}")
        return None, None


def play_audio_file(filename):
    """Play audio from a file."""
    data, samplerate = sf.read(filename, dtype=_audio_dtype)
    logger.info(f"Playing {filename}")
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is finished playing
    logger.info("Playback finished")


def play_audio(audio_data, sampling_rate):
    # audio_segment = AudioSegment(
    #     audio_data.tobytes(),
    #     frame_rate=sampling_rate,
    #     sample_width=audio_data.dtype.itemsize,
    #     channels=1
    # )

    # # Play the audio
    # playback.play(audio_segment)
    sd.play(audio_data, sampling_rate)
    sd.wait()
