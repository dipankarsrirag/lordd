import argparse
import os
import pathlib

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
import torch

import pandas as pd
from datasets import Dataset

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
dtype = torch.bfloat16


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_prompt(data, model="mistral"):
    if model == "mistral":
        prompt = """<s> [INST] From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} [/INST] {}</s>"""
    elif model == "gemma":
        prompt = """<start_of_turn>user From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} <end_of_turn>\n<start_of_turn>model {} <end_of_turn>"""
    else:
        raise NotImplementedError

    prompts = []
    for transcript, word in zip(data["transcript"], data["target_word"]):
        prompts.append(prompt.format(transcript, word))
    return prompts


def get_model_tokenizer(
    model_id,
    attn_implementation,
    use_lora,
    task_type,
    lora_alpha,
    lora_r,
    lora_dropout,
    lora_bias,
):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        print("Return a PEFT Model")
        modules = find_all_linear_names(model)
        print(modules)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias=lora_bias,
            target_modules=modules,
            task_type=task_type,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer, peft_config


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("model_id", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("attn_implementation", type=str)
    parser.add_argument("use_lora", action="store_true")
    parser.add_argument("task_type", type=str)
    parser.add_argument("lora_alpha", type=float)
    parser.add_argument("lora_r", type=float)
    parser.add_argument("lora_dropout", type=float)
    parser.add_argument("lora_bias")
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("train_path", type=pathlib.Path)
    parser.add_argument("valid_path", type=pathlib.Path)
    parser.add_argument("adaptor_dir", type=pathlib.Path)
    parser.add_argument("cache_dir", type=pathlib.Path)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    def get_data(train_path, valid_path):

        train = pd.read_json(train_path)
        valid = pd.read_json(valid_path)

        train["prompt"] = get_prompt(data=train, model=args.model_name)
        valid["prompt"] = get_prompt(data=valid, model=args.model_name)

        train = Dataset.from_pandas(train[["prompt"]], split="train")
        valid = Dataset.from_pandas(valid[["prompt"]], split="test")

        return train, valid

    model, tokenizer, peft_config = get_model_tokenizer(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        use_lora=args.use_lora,
        task_type=args.task_type,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
    )

    train, valid = get_data(path=args.data_path, batch_size=args.batch_size)

    args = TrainingArguments(
        output_dir=args.cache_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=valid,
        dataset_text_field="prompt",
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(3, 0.1)],
    )

    trainer.train()
    trainer.model.save_pretrained(args.adaptor_dir)


if __name__ == "__main__":
    main()
