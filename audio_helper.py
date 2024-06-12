import sounddevice as sd
import soundfile as sf
from pathlib import Path
import logging
import numpy as np


logger = logging.getLogger(__name__)


_audio_dtype = np.float32


class AudioRecorder:
    def __init__(self, samplerate, blocksize, threshold, silence_duration):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.audio_buffer = []
        self.silence_blocks = 0
        self.required_silence_blocks = int(
            silence_duration * samplerate / blocksize)
        self.continue_recording = True

    def is_voice(self, block):
        return np.any(np.abs(block) > self.threshold)

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_voice(indata):
            self.silence_blocks = 0
            # Use only the first channel
            self.audio_buffer.extend(indata[:, 0])
        else:
            self.silence_blocks += 1
            if self.silence_blocks >= self.required_silence_blocks:
                self.continue_recording = False

    def record_until_silence(self):
        self.audio_buffer = []
        self.silence_blocks = 0
        self.continue_recording = True

        with sd.InputStream(callback=self.callback,
                            channels=1,
                            samplerate=self.samplerate,
                            blocksize=self.blocksize,
                            dtype='float32'):
            while self.continue_recording:
                sd.sleep(100)

        return np.array(self.audio_buffer, dtype='float32')


def record_audio(file_output=Path('my_audio.wav'),
                 samplerate=16000,
                 blocksize=1024,
                 threshold=0.25,
                 silence_duration=1.0):
    recorder = AudioRecorder(samplerate, blocksize,
                             threshold, silence_duration)
    logger.info("Recording...")
    recorded_audio = recorder.record_until_silence()
    logger.info("Recording complete.")

    # TODO: send the audio data directly instead of writing the file
    sf.write(file_output, recorded_audio, samplerate)
    logger.info(f"Audio saved to {file_output}.")
    return file_output


def play_audio_file(filename):
    """Play audio from a file."""
    data, samplerate = sf.read(filename, dtype=_audio_dtype)
    logger.info(f"Playing {filename}")
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is finished playing
    logger.info("Playback finished")


def play_audio(audio_data, sampling_rate):
    # Play the audio
    sd.play(audio_data, sampling_rate)
    sd.wait()
