import sys

import torch
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer

LOAD_8BIT = False
finetune_model = "D:\python\LLM_Helper\ds\saved_models\gpt2"
# finetune_model = "D:\huggingface\gpt2"

model = GPT2LMHeadModel.from_pretrained(
    finetune_model,
    load_in_8bit=LOAD_8BIT,
    torch_dtype=torch.float16,
    device_map="auto")

tokenizer = GPT2Tokenizer.from_pretrained(finetune_model)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
):
    def generate_prompt(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""

    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def do_eval():
    while 1:
        instruction = input()
        response = evaluate(instruction)
        if response[-13:] == "<|endoftext|>":
            response = response[:-13]
        print("Response:", response)


def ori_evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
):
    inputs = tokenizer(instruction, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.strip()


def do_ori_eval():
    while 1:
        instruction = input()
        response = ori_evaluate(instruction)
        print("Response:", response)


if __name__ == '__main__':
    do_eval()
    # do_ori_eval()
