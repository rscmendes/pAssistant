import argparse
import logging
import sys

from audio_helper import record_audio, play_audio
from stt import Speech2Text
from tts import Text2Speech
from llm import LLM


def _set_logger():
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s:\n%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    return logger


def _parse_args():
    parser = argparse.ArgumentParser(prog='Privacy Assistant',
                                     description='A privacy assistant with a twist')

    return parser.parse_args()


def _init_models():
    llm = LLM()
    s2t = Speech2Text()
    t2s = Text2Speech()
    return llm, s2t, t2s


def _should_stop(user_query):
    lower_query = user_query.lower()
    return (lower_query.startswith("stop") or "bye bye" in lower_query or "bye-bye" in lower_query)


def main():
    logger = _set_logger()
    # args = _parse_args()

    logger.info("Initializing the models, wait patiently...")
    llm, s2t, t2s = _init_models()
    logger.info("Done.")

    history = ''
    while True:
        # TODO pass the data directly instead of file
        # audio_file = 'prompt.wav'  # for testing
        logger.info("Starting to record. Ask your question now.")
        audio_file = record_audio()
        # play_audio_file(audio_file)

        user_query = s2t.speech2text(audio_file)
        logger.info(f"Query: {user_query}")

        audio_file.unlink()

        # user_query = "What is 1 + 2?"  # for testing
        assistant_response, history = llm.get_response(user_query, history)

        logger.info(f"Response: {assistant_response}")

        logger.debug(f"History: {history}")

        # assistant_response = "The capital of Germany is Berlin"  # testing
        assistant_audio, sampling_rate = t2s.text2speech(assistant_response)
        play_audio(assistant_audio, sampling_rate)

        if _should_stop(user_query):
            break


if __name__ == "__main__":
    main()
