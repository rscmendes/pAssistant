from pathlib import Path
import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration, \
    BitsAndBytesConfig

from audio_helper import load_audio_file


class Speech2Text():
    def __init__(self, model_name: str = "openai/whisper-base") -> None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            device_map='auto')
        self._device = next(iter(self._model.hf_device_map.values()))

        self._processor = WhisperProcessor.from_pretrained(model_name)

    def speech2text(self, audio_file: Path):

        speech, sample_rate = load_audio_file(audio_file)

        # Preprocess the audio file
        input_features = self._processor(
            speech, sampling_rate=sample_rate, return_tensors="pt").input_features  # type: ignore
        input_features = input_features.to(self._device, dtype=torch.float16)

        # Set to EN
        forced_decoder_ids = self._processor.get_decoder_prompt_ids(  # type: ignore
            language="en", task="transcribe")

        # Perform inference
        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids)  # type: ignore

        # Decode the predicted tokens
        transcription = self._processor.decode(  # type: ignore
            predicted_ids[0], skip_special_tokens=True)  # type: ignore

        return transcription
