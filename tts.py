import torch
from transformers import SpeechT5Processor, \
    SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


class Text2Speech():
    def __init__(self,
                 model_name: str = "microsoft/speecht5_tts",
                 vocoder_name: str = "microsoft/speecht5_hifigan",
                 processor_name: str = "microsoft/speecht5_tts",
                 samplerate: int = 16000) -> None:
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self._model.to(self._device)

        self._vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
        self._vocoder.to(self._device)

        # TODO replace with simpler approach, not hardcoded
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation")
        self._speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self._device)

        self._processor = SpeechT5Processor.from_pretrained(processor_name)

        self._samplerate = samplerate

    def text2speech(self, text: str):
        inputs = self._processor(
            text=text, return_tensors="pt").to(self._device)
        speech = self._model.generate_speech(
            inputs["input_ids"], self._speaker_embeddings, vocoder=self._vocoder)

        return speech.cpu().numpy(), self._samplerate
