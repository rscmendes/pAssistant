from pathlib import Path
import numpy as np
from pydub import AudioSegment
import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration, \
    BarkModel, BarkProcessor


_framerate = 16000
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_default_stt_model = "openai/whisper-base"
_stt_model = None
_stt_processor = None


_default_tts_model = "suno/bark"
_tts_model = None
_tts_processor = None


def _load_audio(file_path):
    # Convert the audio file to the correct format using pydub
    audio = AudioSegment.from_file(
        file_path).set_channels(1).set_frame_rate(_framerate)
    # Normalize audio to [-1, 1]
    samples = torch.tensor(audio.get_array_of_samples()).float() / 32768.0
    return samples, _framerate


def speech2text(audio_file: Path, model_name=_default_stt_model):
    # lazy loading
    global _stt_model, _stt_processor
    if _stt_model is None:
        _stt_model = WhisperForConditionalGeneration.from_pretrained(
            model_name)
        _stt_model.to(_device)

    if _stt_processor is None:
        _stt_processor = WhisperProcessor.from_pretrained(model_name)

    speech, sample_rate = _load_audio(audio_file)

    # Preprocess the audio file
    input_features = _stt_processor(
        speech, sampling_rate=sample_rate, return_tensors="pt").input_features  # type: ignore
    input_features = input_features.to(_device)

    # Set to EN
    forced_decoder_ids = _stt_processor.get_decoder_prompt_ids(  # type: ignore
        language="en", task="transcribe")

    # Perform inference
    with torch.no_grad():
        predicted_ids = _stt_model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids)  # type: ignore

    # Decode the predicted tokens
    transcription = _stt_processor.decode(  # type: ignore
        predicted_ids[0], skip_special_tokens=True)  # type: ignore

    return transcription


def text2speech(text):
    global _tts_model, _tts_processor
    if _tts_model is None:
        _tts_model = BarkModel.from_pretrained(_default_tts_model)
        _tts_model.to(_device)

    if _tts_processor is None:
        _tts_processor = BarkProcessor.from_pretrained(_default_tts_model)

    inputs = _tts_processor(text, return_tensors="pt")

    # the attention mask was resulting in very strange behavior
    # inputs['attention_mask'] = torch.ones(
    #     inputs.input_ids.shape, dtype=torch.long).to(_device)
    inputs = {key: val.to(_device) for key, val in inputs.items()}
    inputs['pad_token_id'] = _tts_processor.tokenizer.pad_token_id

    speech_values = _tts_model.generate(
        **inputs, do_sample=True).cpu().numpy().squeeze()
    sampling_rate = _tts_model.generation_config.sample_rate
    return (speech_values * 32767).astype(np.int16), sampling_rate
