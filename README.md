# Personal assistant
This is a personal assistant with a twist: a strong sarcastic personality. He might refuse to answer or provide witty and humorous responses.

This is a pet project to exploit the great HuggingFace libraries and available open-source models. My objective was to run the personal assistant in the GPU and fully local, with I achieved with just 6 GB of GPU memory.

Current model stack:
- microsoft/Phi-3.5-mini-instruct as LLM
- openai/whisper-base as the speech-to-text model
- microsoft/speecht5_tts as the text-to-speech model
- "microsoft/speecht5_hifigan" as vocoder.
Phi-3.5 and Whisper were quantized to 4 bits to fit my GPU.

# WIP log
Speech to text models tried:
* Whisper: great quality but it is very slow. Unfortunately, it does not fit in my GPU, and quantization is only supported on CPU.
    * Changed to Whisper-base.
* facebook/wav2vec2-large-960h: the transcription is with very bad quality, I was not able to make it work. 
* deepspeech not supported with my stack -- didn't check why, but pip can't satisfy requirements

Text to speech models tried:
* suno/bark: uses almost all my GPU resources. Works most of the times, but sometimes it gives random output.

TODO:
- [x] Microphone should be active only while talking, instead of using a fixed duration.
- [ ] Pass audio data directly to model, instead of file. For some reason this was crashing the model.
- [x] Use a quantized version of Phi-3? The problem is that the text2speech model is already using all GPU. But I can maybe move from GPU to CPU while idle and vice-versa when using it.
- [x] Quantize whisper
- [x] Bark does not allow for quantization, and takes about 5GB in GPU. Replace with other model. 
    - Replaced with SpeechT5. Also doesn't allow quantization, but is significantly smaller. It also allows for the Vocoder integration.
- [x] Audio cuts for big answers.
    - Fixed with SpeechT5
- [ ] Improve the vocoder to allow selecting the speaker
- [x] Add the history to the model, to allow continuous chat.
- [x] Initialize the models at startup, instead of lazy loading
- [ ] Clean code

