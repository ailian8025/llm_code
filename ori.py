import os
import sys
import copy
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from dataclasses import dataclass, field
from datasets import load_dataset
import transformers
from collections import namedtuple

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GenerationConfig

model_path = "D:\huggingface\gpt2"
per_gpu_train_batch_size = 4
gradient_accumulation_steps = 16
epochs = 3
learning_rate = 3e-4
cutoff_len = 512
val_set_size = 50
data_path = "D:\datasets\Alpaca-CoT\Vicuna_5000.json"
world_size = 1
IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def generate_prompt(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    return prompt_.format_map(data_point)


def get_data_model():
    data = load_dataset("json", data_files=data_path)

    model = GPT2LMHeadModel.from_pretrained(model_path).half()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # TODO 需要确定！
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = tokenizer.eos_token
    return data, model, tokenizer


def train():
    # 1. load data & model_class
    data, model, tokenizer = get_data_model()

    def tokenize(prompt):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=cutoff_len,
                           #    padding="max_length",
                           padding=False,
                           )

        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "labels": copy.deepcopy(result["input_ids"])
        }

    def generate_and_tokenize_prompt(data_point):
        prompt_no_resp = generate_prompt(data_point)
        tokenized_result = tokenize(prompt_no_resp)
        source_len = len(tokenized_result['input_ids'])

        prompt_with_response = prompt_no_resp + " " + data_point["output"]
        # 需要自定义添加
        prompt_with_response += " " + tokenizer.eos_token

        tokenized_with_response = tokenize(prompt_with_response)

        tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]

        return tokenized_with_response

    model_name = model_path.split('\\')[-1]
    output_dir = f"temp_saved_models_half/{model_name}"

    # 2. split dataset
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # 3. train
    total_batch_size = per_gpu_train_batch_size * gradient_accumulation_steps * world_size
    total_optim_steps = train_data.num_rows // total_batch_size

    print("***** Running training *****")
    print(f"  Num Epochs = {epochs}", )
    print(f"  Instantaneous batch size per GPU = {per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {total_optim_steps}")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_gpu_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            report_to='tensorboard'
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
