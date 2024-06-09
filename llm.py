import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_default_llm = "microsoft/Phi-3-mini-4k-instruct"
_model = None
_tokenizer = None

# Not enough GPU memory
# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_device = "cpu"


def _pre_process_prompt(prompt):
    # TODO add history
    return f"<|user|>\n{prompt} <|end|>\n<|assistant|>"


def get_response(prompt, model_name=_default_llm):
    # lazy loading
    global _model, _tokenizer
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        _model.to(_device)

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = _tokenizer(_pre_process_prompt(prompt),
                        return_tensors="pt").to(_device)
    outputs = _model.generate(inputs['input_ids'], max_length=100)
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the generated text
    input_length = len(_tokenizer.decode(
        inputs['input_ids'][0], skip_special_tokens=True))

    return response[input_length:].strip()
