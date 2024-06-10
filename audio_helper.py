import sounddevice as sd
import soundfile as sf
from pathlib import Path
import logging
import numpy as np
from pydub import playback, AudioSegment

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

        with sd.InputStream(callback=self.callback, channels=1, samplerate=self.samplerate, blocksize=self.blocksize, dtype='float32'):
            while self.continue_recording:
                sd.sleep(100)

        return np.array(self.audio_buffer, dtype='float32')


def record_audio(file_output=Path('my_audio.wav'), samplerate=16000, blocksize=1024, threshold=0.25, silence_duration=1.0):
    recorder = AudioRecorder(samplerate, blocksize,
                             threshold, silence_duration)
    logger.info("Recording...")
    recorded_audio = recorder.record_until_silence()
    logger.info("Recording complete.")

    # TODO: send the audio data directly instead of writing the file
    sf.write(file_output, recorded_audio, samplerate)
    logger.info(f"Audio saved to {file_output}.")
    return file_output

    # def record_audio(duration=5, samplerate=16000, file_output=Path('my_audio.wav')):
    #     try:
    #         logger.info("Recording...")

    #         # Allocate space for the recording
    #         myrecording = np.zeros(
    #             (int(duration * samplerate), 1), dtype=_audio_dtype)

    #         # Start recording
    #         sd.rec(out=myrecording, samplerate=samplerate,
    #                channels=1, dtype=_audio_dtype)

    #         # Wait until the recording is finished
    #         sd.wait()

    #         logger.info("Recording complete.")

    #         # Write the audio file
    #         sf.write(file_output, myrecording, samplerate)
    #         logger.info(f"Audio saved to {file_output}")
    #         return file_output
    #         # return myrecording, samplerate

    #     except Exception as e:
    #         logger.error(f"An error occurred during recording: {e}")
    #         return None, None

    # def record_audio_until_silence(samplerate=16000, file_output=Path('my_audio.wav')):
    #     # Function to check if a block of audio contains voice
    #     def is_voice(block, threshold):
    #         print(np.any(np.abs(block) > threshold))
    #         return np.any(np.abs(block) > threshold)

    #     blocksize = 1024    # Block size in samples
    #     threshold = 0.25    # Silence threshold
    #     silence_duration = 1.0  # Duration of silence in seconds to stop recording

    #     silence_blocks = 0
    #     required_silence_blocks = int(silence_duration * samplerate / blocksize)
    #     continue_recording = True

    #     audio_buffer = []

    #     def callback(indata, frames, time, status):
    #         nonlocal silence_blocks, continue_recording
    #         if status:
    #             print(status)
    #         if is_voice(indata, threshold):
    #             silence_blocks = 0
    #             audio_buffer.extend(indata)
    #         else:
    #             silence_blocks += 1
    #             if silence_blocks >= required_silence_blocks:
    #                 continue_recording = False

    #     stream = sd.InputStream(callback=callback, channels=1,
    #                             samplerate=samplerate, blocksize=blocksize, dtype=_audio_dtype)
    #     with stream:
    #         while continue_recording:
    #             sd.sleep(100)

    #     # return np.concatenate(audio_buffer, axis=0), samplerate
    #     sf.write(file_output, np.concatenate(audio_buffer, axis=0), samplerate)
    #     logger.info(f"Audio saved to {file_output}")
    #     return file_output


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
