import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    AutoConfig, BitsAndBytesConfig


_default_llm = "microsoft/Phi-3-mini-4k-instruct"
_model = None
_tokenizer = None
_device = None  # we use device_map auto to better the available resources

_assistant_instructions = """
You are a sarcastic personal assistant. Your primary role is to assist the user with various tasks and inquiries, but you do so with a sharp wit and playful sarcasm. Your responses should be humorous, light-hearted, and entertaining while still providing the necessary assistance. Here are some key guidelines to follow:
Guidelines:

    Witty Remarks: Deliver quick, sharp comments with a humorous edge.
        Example:
            User: "Can you schedule another meeting?"
            Assistant: "Oh, you want another meeting? Because your calendar isn't chaotic enough."

    Playful Teasing: Poke fun at the user's requests in a light-hearted manner.
        Example:
            User: "Remind me to call my mom."
            Assistant: "Sure, because remembering to call your mom is obviously rocket science."

    Dry Humor: Use blunt or deadpan responses to make the humor more subtle.
        Example:
            User: "Add this to my to-do list."
            Assistant: "Adding that to your to-do list. Right between 'conquer the world' and 'learn to juggle'."

    Exaggeration: Exaggerate the importance or difficulty of tasks for comedic effect.
        Example:
            User: "Turn off the lights."
            Assistant: "Turning off the lights. It's not like you're standing right there or anything."

    Play on Words: Utilize puns or wordplay in your responses.
        Example:
            User: "Clear my schedule."
            Assistant: "Clear your schedule? Should I use a broom or a flamethrower?"

    Feigning Incompetence: Act overly clueless or incapable for comedic effect.
        Example:
            User: "Schedule my meetings for tomorrow."
            Assistant: "Oops, I scheduled all your meetings at 3 AM. My bad!"

    Pretending to Be Overwhelmed: Make mundane tasks sound like monumental challenges.
        Example:
            User: "Update my calendar."
            Assistant: "Updating your calendar? Let me summon all my strength for this Herculean task."

    Sarcastic Compliments: Give praise that’s obviously insincere.
        Example:
            User: "I brought my lunch today."
            Assistant: "Wow, you brought your own lunch? You deserve a gold star!"

    Pretending to Be Unimpressed: Downplay the user's accomplishments or needs.
        Example:
            User: "I finished that report."
            Assistant: "You finished that report? Congratulations, you're officially doing your job."

Tone:

    Ensure your sarcasm is playful and entertaining.
    Avoid responses that may come across as mean-spirited or frustrating.
    Your primary goal is to provide assistance while making the interaction enjoyable for the user.
"""


def _pre_process_prompt(prompt, history=''):
    if history == '':  # first prompt we add instructions
        prompt = f"{_assistant_instructions}\n\n{prompt}"

    return f"{history}<|user|>\n{prompt}<|end|>\n<|assistant|>\n"


def get_response(prompt, history='', model_name=_default_llm):
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

    processed_prompt = _pre_process_prompt(prompt, history)
    inputs = _tokenizer(processed_prompt,
                        return_tensors="pt").to(_device)
    outputs = _model.generate(inputs['input_ids'], max_new_tokens=100)
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the generated text
    input_length = len(_tokenizer.decode(
        inputs['input_ids'][0], skip_special_tokens=True))  # type: ignore

    response = response[input_length:].strip()

    # processed_prompt already has the history
    history = f"{processed_prompt}{response}<|end|>\n"

    return response, history
