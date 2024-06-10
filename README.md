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
- [ ] Quantize whisper
- [ ] Audio cuts for big answers.

