from pathlib import Path
import numpy as np
from pydub import AudioSegment
import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration, \
    BitsAndBytesConfig, AutoModel, AutoProcessor


_framerate = 16000


_default_stt_model = "openai/whisper-base"
_stt_model = None
_stt_processor = None
_stt_device = None  # we use device_map auto to better use the resources


_default_tts_model = "suno/bark"
_tts_model = None
_tts_processor = None
_tts_device = None  # we use device_map auto to better use the resources


def _load_audio(file_path):
    # Convert the audio file to the correct format using pydub
    audio = AudioSegment.from_file(
        file_path).set_channels(1).set_frame_rate(_framerate)
    # Normalize audio to [-1, 1]
    samples = torch.tensor(audio.get_array_of_samples()).float() / 32768.0
    return samples, _framerate


def speech2text(audio_file: Path, model_name=_default_stt_model):
    # lazy loading
    global _stt_model, _stt_processor, _stt_device
    if _stt_model is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        _stt_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            device_map='auto')
        _stt_device = next(iter(_stt_model.hf_device_map.values()))

    if _stt_processor is None:
        _stt_processor = WhisperProcessor.from_pretrained(model_name)

    speech, sample_rate = _load_audio(audio_file)

    # Preprocess the audio file
    input_features = _stt_processor(
        speech, sampling_rate=sample_rate, return_tensors="pt").input_features  # type: ignore
    input_features = input_features.to(_stt_device, dtype=torch.float16)

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
    global _tts_model, _tts_processor, _tts_device
    if _tts_model is None:
        # The BarkModel does not allow quantization nor device_map
        _tts_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        _tts_model = AutoModel.from_pretrained(_default_tts_model)
        _tts_model.to(_tts_device)

    if _tts_processor is None:
        _tts_processor = AutoProcessor.from_pretrained(_default_tts_model)

    inputs = _tts_processor(text, return_tensors="pt")

    # the attention mask was resulting in very strange behavior
    # inputs['attention_mask'] = torch.ones(
    #     inputs.input_ids.shape, dtype=torch.long).to(_device)
    inputs = {key: val.to(_tts_device) for key, val in inputs.items()}
    inputs['pad_token_id'] = _tts_processor.tokenizer.pad_token_id

    speech_values = _tts_model.generate(
        **inputs, do_sample=True).cpu().numpy().squeeze()
    sampling_rate = _tts_model.generation_config.sample_rate
    return (speech_values * 32767).astype(np.int16), sampling_rate
