import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    AutoConfig, BitsAndBytesConfig


_default_llm = "microsoft/Phi-3-mini-4k-instruct"
_model = None
_tokenizer = None
_device = None  # we use device_map auto to better the available resources


def _pre_process_prompt(prompt):
    # TODO add history
    return f"<|user|>\n{prompt} <|end|>\n<|assistant|>"


def get_response(prompt, model_name=_default_llm):
    # lazy loading
    global _model, _tokenizer, _device
    if _model is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )
        _device = next(iter(_model.hf_device_map.values()))

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = _tokenizer(_pre_process_prompt(prompt),
                        return_tensors="pt").to(_device)
    outputs = _model.generate(inputs['input_ids'], max_length=100)
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the generated text
    input_length = len(_tokenizer.decode(
        inputs['input_ids'][0], skip_special_tokens=True))  # type: ignore

    return response[input_length:].strip()
