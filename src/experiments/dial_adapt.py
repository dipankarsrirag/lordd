import argparse
import os
import warnings
from tqdm import tqdm
import pathlib

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# -----------------
# Global constants
# -----------------
MAX_LEN = 512
DTYPE = torch.bfloat16


# -----------------
# Helper functions
# -----------------
def set_env():
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # plural


def find_all_linear_names(model):
    """
    Find all 4-bit linear modules to target with LoRA.
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split(".")
            lora_module_names.add(parts[-1] if len(parts) > 0 else name)
    return sorted(list(lora_module_names))


def get_prompt(data, label="original", model="mistral"):
    if model == "mistral":
        prompt = (
            "<s> [INST] From the conversation, Replace \"[MASK]\" with the most relevant word. "
            "Generate a single token, do not give explanation.\nConversation:{} [/INST]"
        )
    elif model == "gemma":
        prompt = (
            "<start_of_turn>user From the conversation, Replace \"[MASK]\" with the most relevant word. "
            "Generate a single token, do not give explanation.\nConversation:{} <end_of_turn>"
        )
    else:
        raise NotImplementedError(f"Unknown model prompt style: {model}")
    return [prompt.format(text) for text in data[label]]


def gather_mask_vec_from_last_layer(hidden_states, input_ids, attention_mask, tokenizer, reduce="mean"):
    """
    hidden_states: tuple length L; each [B, T, H]
    input_ids: [B, T] LongTensor
    attention_mask: [B, T] LongTensor
    Returns: [B, H] pooled at positions where token == mask_token_id.
    If absent, falls back to last real token (by attention_mask).
    If multiple [MASK] tokens exist, pools across them (mean/first/max).
    """
    last = hidden_states[-1]  # [B, T, H]
    B, T, H = last.shape
    reps = []
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    for b in range(B):
        ids = input_ids[b]
        am = attention_mask[b]
        mask_positions = (ids == mask_id).nonzero(as_tuple=False).flatten()

        if mask_positions.numel() > 0:
            vecs = last[b, mask_positions, :]  # [M, H]
            if reduce == "mean":
                vec = vecs.mean(dim=0)
            elif reduce == "first":
                vec = vecs[0]
            elif reduce == "max":
                vec = vecs.max(dim=0).values
            else:
                vec = vecs.mean(dim=0)
        else:
            # Fallback: last non-pad (or last where attention_mask==1)
            if am.any():
                j = (am.nonzero(as_tuple=False).flatten())[-1].item()
            else:
                # last non-pad id
                non_pad = (ids != pad_id).nonzero(as_tuple=False).flatten()
                j = non_pad[-1].item() if non_pad.numel() > 0 else 0
            vec = last[b, j, :]

        reps.append(vec)

    return torch.stack(reps, dim=0)  # [B, H]


def get_model_tokenizer(
    model_id: str,
    attn_implementation: str = None,
    use_lora: bool = False,
    task_type: str = "CAUSAL_LM",
    lora_alpha: float = 16,
    lora_r: int = 8,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",
    prefer_causal_lm: bool = True,
):
    """
    Load a 4-bit model + tokenizer. Optionally wrap with LoRA.
    Returns model, tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )

    ModelClass = AutoModelForCausalLM if prefer_causal_lm else AutoModel

    model = ModelClass.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=DTYPE,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        print("Wrapping model with PEFT LoRA adapters...")
        modules = find_all_linear_names(model)
        print("LoRA target modules:", modules)

        # Prepare for k-bit finetuning
        if hasattr(model, "gradient_checkpointing_enable"):
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
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model, tokenizer


def tokenize_text_factory(tokenizer, max_len, field_name):
    def _fn(examples):
        res = tokenizer(
            examples[field_name],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        return res
    return _fn


def compute_original_embeddings_factory(model, tokenizer, batch_size):
    """
    Pre-compute the [B,H] anchor vectors for the 'original' prompts at true [MASK] positions.
    """
    def _fn(examples):
        tok = tokenizer(
            examples["original"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors=None,
        )
        N = len(tok["input_ids"])
        original_vecs = []

        # With device_map='auto', keep tensors on CPU and let HF route internally
        for i in range(0, N, batch_size):
            input_ids = torch.tensor(tok["input_ids"][i:i+batch_size])
            attention_mask = torch.tensor(tok["attention_mask"][i:i+batch_size])

            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            vecs = gather_mask_vec_from_last_layer(
                outputs.hidden_states, input_ids, attention_mask, tokenizer, reduce="mean"
            ).to("cpu").to(DTYPE)

            original_vecs.extend(vecs.numpy())

            del input_ids, attention_mask, outputs, vecs
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        examples["original_embedding"] = original_vecs  # each element is [H]
        return examples
    return _fn


def collate_fn(batch):
    """
    Build a batch of tensors. We expect each example to contain:
    - input_ids, attention_mask (for 'transformed' prompts)
    - label (float: +1 or -1)
    - original_embedding: [H]
    """
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.float)
    original_embedding = torch.tensor([ex["original_embedding"] for ex in batch], dtype=DTYPE)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "original_embedding": original_embedding,
    }


def train_contrastive(
    model,
    tokenizer,
    dataset,
    adaptor_dir: str,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 3,
    margin: float = 0.25,
):
    """
    Train with CosineEmbeddingLoss on [MASK] vectors:
    - anchor: precomputed original [B,H]
    - positive/negative: transformed [B,H]
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss(margin=margin, reduction="mean")

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"].to(DTYPE)  # +1 / -1
            original_hidden = batch["original_embedding"].to(DTYPE)  # [B,H]

            outputs = model(
                input_ids=input_ids,               # keep CPU tensors for device_map='auto'
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            transformed_hidden = gather_mask_vec_from_last_layer(
                outputs.hidden_states, input_ids, attention_mask, tokenizer, reduce="mean"
            ).to(DTYPE)

            # Put both on same device as transformed_hidden (typically CPU under auto routing)
            device = transformed_hidden.device
            labels = labels.to(device)
            original_hidden = original_hidden.to(device)

            loss = criterion(transformed_hidden, original_hidden, labels)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch}/{num_epochs} | Average Loss: {avg_loss:.6f}")

        # Early stopping + save best
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            epochs_no_improve = 0
            try:
                model.save_pretrained(adaptor_dir)
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(adaptor_dir)
                print(f"Saved adapter to: {adaptor_dir}")
            except Exception as e:
                print(f"Warning: could not save adapter: {e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print("Training complete.")


def get_data(args, model, tokenizer):
    """
    Reads JSON, builds prompts, tokenizes transformed, precomputes original embeddings.
    """
    df = pd.read_json(args.data_path)
    if "original" not in df.columns or "transformed" not in df.columns or "label" not in df.columns:
        raise ValueError("JSON must contain 'original', 'transformed', and 'label' fields.")

    df["original"] = get_prompt(df, label="original", model=args.model_name)
    df["transformed"] = get_prompt(df, label="transformed", model=args.model_name)

    dataset = Dataset.from_pandas(df)

    # Tokenize 'transformed' prompts -> these will be fed during training
    dataset = dataset.map(
        tokenize_text_factory(tokenizer, MAX_LEN, "transformed"),
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=True,
        desc="Tokenizing transformed prompts",
    )

    # Precompute anchor embeddings for the 'original' prompts
    dataset = dataset.map(
        compute_original_embeddings_factory(model, tokenizer, args.batch_size),
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=True,
        desc="Precomputing original embeddings",
    )

    # Keep only necessary columns
    keep_cols = ["input_ids", "attention_mask", "label", "original_embedding"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])
    return dataset


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive training with true [MASK] token (CosineEmbeddingLoss).")
    p.add_argument("--model_id", type=str, required=True, help="HF model id (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
    p.add_argument("--model_name", type=str, choices=["mistral", "gemma"], default="mistral", help="Prompt style preset")
    p.add_argument("--attn_implementation", type=str, default=None, help="eager | flash_attention_2 | sdpa, etc.")

    # LoRA options
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters")
    p.add_argument("--task_type", type=str, default="CAUSAL_LM", help="PEFT task type, usually CAUSAL_LM")
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_bias", type=str, default="none")

    # Training
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.25)

    # Data / output
    p.add_argument("--data_path", type=pathlib.Path, required=True)
    p.add_argument("--adaptor_dir", type=pathlib.Path, required=True)

    args = p.parse_args()
    return args


def main():
    set_env()
    args = parse_args()

    model, tokenizer = get_model_tokenizer(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        use_lora=args.use_lora,
        task_type=args.task_type,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        prefer_causal_lm=True,
    )

    dataset = get_data(args, model, tokenizer)

    train_contrastive(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        adaptor_dir=str(args.adaptor_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
