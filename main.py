import argparse
import logging
import sys
from dotenv import load_dotenv

from audio_helper import record_audio, play_audio, play_audio_file
from tts import speech2text, text2speech
from llm import get_response


def _set_logger():
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    return logger


def _parse_args():
    parser = argparse.ArgumentParser(prog='Privacy Assistant',
                                     description='A privacy assistant with a twist')

    return parser.parse_args()


def main():
    logger = _set_logger()
    # args = _parse_args()

    user_query = ""
    while True:
        # audio_file = 'prompt.wav'
        # TODO pass the data directly instead of file
        audio_file = record_audio()
        # user_speech, sample_rate = record_audio()
        # play_audio(user_speech, sample_rate)

        user_query = speech2text(audio_file)
        print(f"User: {user_query}")

        audio_file.unlink()

        if user_query.lower().startswith("stop") or "bye bye" in user_query.lower():  # TODO make smarter finish
            break

        # user_query = "What is 1 + 2?"
        assistant_response = get_response(user_query)
        print(f"Assistant: {assistant_response}")
        # assistant_response = "Berlin is great, thank you"

        assistant_audio, sampling_rate = text2speech(assistant_response)
        play_audio(assistant_audio, sampling_rate)


if __name__ == "__main__":
    main()
