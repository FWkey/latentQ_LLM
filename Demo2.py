import os
from threading import Thread
from typing import Iterator

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """\
# finetuned with quantized latent weight. 
"""

LICENSE = """
<p/>
---
As a derivate work of [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) by Meta,
this demo is governed by the original [license](https://huggingface.co/meta-llama/llama-2-7b/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/meta-llama/llama-2-7b/blob/main/USE_POLICY.md).
"""

import os
from utils.quant import QLinearA16W4

def load_quantized_param(model, ckp):
    num_layers = model.config.num_hidden_layers
    decoder_name = ''
    for name, child in model.named_modules():
        if isinstance(child, torch.nn.ModuleList) and len(child) == num_layers:
            decoder_layers = child
            decoder_name = name + '.'
            break
    if not decoder_name:
        assert False, 'decoder not found!'

    if ckp and os.path.exists(ckp):
        checkpoint = ckp
        print('Loading model ...')
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            ckt = safe_load(checkpoint)
        else:
            ckt = torch.load(checkpoint)
    else:
        assert False, 'load_param not exist!'

    i = -1
    for module in list(decoder_layers):
        i += 1
        wbit = 4
        for subname, submodule in list(module.named_modules()):
            for name, child in submodule.named_children():
                if isinstance(child, torch.nn.Linear):
                    if child.weight is None:
                        warnings.warn(f'{child} has empty weight!')
                        continue
                    if wbit == 4:
                        newlayer = QLinearA16W4(child.weight.shape[1], child.weight.shape[0],
                                                bias=child.bias is not None, lbit=4, gbit=16)
                        tsubname = subname + '.' if subname else subname
                        newlayer.lname = decoder_name + str(i) + '.' + tsubname + name
                        newlayer.zeros.data = ckt[decoder_name + str(i) + '.' + tsubname + name + '.zeros'].to(
                            torch.bfloat16)
                        newlayer.scales.data = ckt[decoder_name + str(i) + '.' + tsubname + name + '.scales'].to(
                            torch.bfloat16)
                        newlayer.weight.set_value(ckt[decoder_name + str(i) + '.' + tsubname + name + '.qweight'],
                                                  [newlayer.scales, newlayer.zeros, newlayer.wshape])
                        newlayer.weight.residual.name = decoder_name + str(i) + '.' + tsubname + name + '.residual'
                    else:
                        assert False, 'only support 4bit weight quantization!'
                    if child.bias is not None:
                        newlayer.bias.data = ckt[decoder_name + str(i) + '.' + tsubname + name + '.bias'].to(
                            torch.bfloat16)
                    setattr(submodule, name, newlayer)
                elif hasattr(child, 'weight') and child.weight is not None:
                    tsubname = subname + '.' if subname else subname
                    try:
                        child.weight.data = ckt[decoder_name + str(i) + '.' + tsubname + name + '.weight']
                    except:
                        assert False, f"{decoder_name + str(i) + '+' + tsubname + name} not found!"
                    if hasattr(child, 'bias') and child.bias is not None:
                        child.bias.data = ckt[decoder_name + str(i) + '.' + tsubname + name + '.bias']
    return model, decoder_layers


def model_to_device(model, device):
    model.to(device)

    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, QLinearA16W4):
                child.to(device)

#model_path = "../hf_cache/llama2-13b-chat/"
#checkpoint_dir = '../GPTQ/openllm13bw4.safetensors'
model_path = "./hf_cache/llama2-7b/"
#checkpoint_dir1 = './output4/llamaT7b-gpt4-2.pth'
checkpoint_dir2 = '../GPTQ/llamaT7bw4.safetensors'

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


if torch.cuda.is_available():
    model_id = model_path#"meta-llama/Llama-2-13b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer.use_default_system_prompt = False
    model, decoder = load_quantized_param(model,checkpoint_dir2)
    model.bfloat16()
    model_to_device(model,'cuda')

def convert_history_to_text(start_message,history,message):
    if not start_message:
        start_message = ""
    text = start_message + "".join(
        [
            "".join(
                [
                    f"### Human: {item[0]}\n",
                    f"### Assistant: {item[1]}\n",
                ]
            )
            for item in history
        ]
    )
    text += "".join(
                [
                    f"### User: {message}\n",
                    f"### Response:\n",
                ]
            )
    return text


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = convert_history_to_text(system_prompt,chat_history,message)
    input_ids = tokenizer(conversation, return_tensors="pt").input_ids
    #if system_prompt:
     #   conversation.append({"role": "system", "content": system_prompt})
    #for user, assistant in chat_history:
     #   conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    # conversation.append({"role": "user", "content": message})

    #input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
    print("".join(outputs))

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6, value="""You are an artificial intelligence assistant that gives helpful, detailed, and polite response to the human user. You are served as a chatbot to give a Response to the User. """),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.05,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=20,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.3,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Give three tips for staying healthy."],
        ["Why to follow the first tip?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
["Summarize this sentence: \"When buying furniture, it's important to take into account the overall look and feel of the space. That means that you should ask yourself if the pieces you are selecting are a good match for the existing furnishings as well as what you imagine the overall style to be.\""]
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
