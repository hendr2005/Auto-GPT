import openai
from config import Config
import os

from llm_models import VicunaModel


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"
cfg = Config()

if cfg.use_vicuna:
    import torch
    from fastchat.conversation import SeparatorStyle
else:
    torch = None
    SeparatorStyle = None


openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_vicuna:
        return create_vicuna_completions(messages)
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


def create_vicuna_completions(messages):
    results = [vicuna_interact(message) for message in messages]
    if cfg.debug:
        print("results", [results])
    result = None
    for result in results:
        if isinstance(result, list):
            for item in result:
                if item.startswith("{"):
                    result = item
                    break
    if cfg.debug:
        print("result", result)
    return result


def vicuna_interact(message, temperature=0.7, max_new_tokens=512):
    # model_name = args.model_name
    instance = VicunaModel()
    model = instance.model
    tokenizer = instance.tokenizer
    # Chat
    if cfg.debug:
        print(message)
    conv = instance.conv
    role = message['role']
    if role == 'system':
        if conv.system == "":
            conv.system = message['content']
            return ""
        if message['content'].startswith("Permanent"):
            conv.append_message(conv.roles[0], message['content'])
    if role == 'user':
        conv.append_message(conv.roles[1], message['content'])
    if role == 'assistant':
        if message['content'] == '' or len(message['content']) < 10:
            return ""
    #    conv.append_message(conv.roles[2], message['content'])
    #if role == 'gpt':
    #    conv.append_message(conv.roles[3], message['content'])
    conv.append_message(conv.roles[3], None)
    generate_stream_func = generate_stream
    prompt = conv.get_prompt()
    skip_echo_len = len(prompt) + 1
    # print(prompt)
    params = {
        "model": cfg.vicuna_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }
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
    conv.messages[-1][-1] = output

    if cfg.debug:
        print([output])
    return output
