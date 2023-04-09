import openai
from config import Config
from typing import Any, Tuple
import os
import sys
import time
import json
from data import load_prompt
from pathlib import Path
import argparse
import time




from config import Singleton
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"
cfg = Config()

if cfg.use_vicuna:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LLaMATokenizer, AutoModel

    from fastchat.conversation import conv_templates, SeparatorStyle
    from fastchat.serve.compression import compress_module
    from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
else:
    torch = None
    fire = None
    initialize_model_parallel = None
    ModelArgs = None
    Transformer = None
    Tokenizer = None
    LLaMA = None
    BetterTransformer = None
    AutoModelForSequenceClassification = None


openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_llama:
        return create_llama_completeion(messages) #, temperature, max_tokens)
    elif cfg.use_vicuna:
        return create_vicuna_completions(messages) #, temperature, max_tokens)
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]


def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.to("cuda")
    elif device == "mps":
        model.to("mps")

    if (device == "mps" or device == "cpu") and load_8bit:
        compress_module(model)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 0.7))
    max_new_tokens = int(params.get("max_new_tokens", 512))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values


class VicunaModel(metaclass=Singleton):
    def __init__(self, max_tokens=512, max_batch_size=32):
        
        model_name = cfg.vicuna_path
        llm_device = cfg.llm_device
        num_gpus = 1  # cfg.num_gpus
        load_8bit = False  # cfg.load_8bit
        self.conv = conv_templates["bair_v1"].copy()
        self.conv.roles = ("system", "user", 'assistant', 'gpt')
        self.conv.sep = "###"
        self.conv.sep_style = SeparatorStyle.SINGLE
        self.conv.system = ""
        print("Loading Vicuna model...")

    # Model
        model, tokenizer = load_model(
            model_name, llm_device,
            num_gpus, load_8bit, False
        )
        self.model = model
        self.tokenizer = tokenizer
        print("Loaded Vicuna model!")


def create_vicuna_completions(messages):
    return [vicuna_interact(message) for message in messages][0]


def vicuna_interact(message, temperature=0.7, max_new_tokens=512):
    # model_name = args.model_name
    instance = VicunaModel()
    model = instance.model
    tokenizer = instance.tokenizer
    # Chat
    print(message)
    conv = instance.conv
    role = message['role']
    if role == 'system':
        if conv.system == "":
            conv.system = message['content']
            return ""
        conv.append_message(conv.roles[0], message['content'])
        # return ""
    if role == 'user':
        conv.append_message(conv.roles[1], message['content'])
    if role == 'assistant':
        conv.append_message(conv.roles[2], message['content'])
    if role == 'gpt':
        conv.append_message(conv.roles[3], message['content'])
    conv.append_message(conv.roles[3], None)
    generate_stream_func = generate_stream
    prompt = conv.get_prompt()
    skip_echo_len = len(prompt) + 1
    print(prompt)
    params = {
        "model": cfg.vicuna_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }

    print(f"{conv.roles[3]}: ", end="", flush=True)
    pre = 0
    for outputs in generate_stream_func(model, tokenizer, params, cfg.llm_device):
        outputs = outputs[skip_echo_len:].strip()
        outputs = outputs.split(" ")
        now = len(outputs)
        if now - 1 > pre:
            print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
            pre = now - 1
    # print(" ".join(outputs[pre:]), flush=True)
    output =  " ".join(outputs)
    conv.messages[-1][-1]  = output

    if True:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return output
