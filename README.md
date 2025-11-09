# Parameter-Efficient Fine-Tuning on FLAN-T5
Compare Prompt-Tuning, Prefix-Tuning, and LoRA on two tasks (classification and summarization) using the FLAN-T5 family. Your objective is to measure how performance improves as you increase only a small number of trainable parameters, and to plot parameter-efficiency vs performance curves.

Why FLAN-T5?
- FLAN-T5 is instruction-tuned T5. It works well as a single text-to-text model for both classification and summarization.
- Variants: google/flan-t5-small (77M), base (250M), large (780M), xl, xxl. Start with small/base for student hardware.

---

## TL;DR: What you’ll build
- A reproducible pipeline to fine-tune FLAN-T5 with three PEFT methods:
  - Prompt-Tuning (learn soft prompt tokens)
  - Prefix-Tuning (learn prefix key/value states)
  - LoRA (insert low-rank adapters in attention)
- Two downstream tasks:
  - Classification: SST-2 sentiment (binary)
  - Summarization: SAMSum (dialogue → summary)
- Plots of “capacity (trainable params) vs performance (Accuracy/F1 or ROUGE)”.

You will freeze FLAN-T5’s weights and only train the adapters.

---

## Tasks and datasets
- Classification: GLUE SST-2
  - Input example: “sst2: the plot was predictable but the acting was good”
  - Target: “positive” or “negative”
- Summarization: SAMSum
  - Input: multi-turn dialogue transcript
  - Target: a concise summary

Both are available via Hugging Face Datasets.

---

## Metrics and curves
- Classification: Accuracy, F1 (positive as the positive class)
- Summarization: ROUGE-1/2/L (report at least ROUGE-L)
- Parameter-efficiency: number of trainable parameters in the adapters
- Curves:
  - X-axis: capacity (LoRA rank r OR number of virtual tokens)
  - Y-axis: metric (F1 for classification, ROUGE-L for summarization)
  - One curve per method, one plot per task
  - Optionally log-scale the X-axis

---

## Environment setup
Install dependencies (FLAN-T5 uses SentencePiece tokenization):
```bash
pip install -U "transformers>=4.44" "datasets>=2.20" "peft>=0.12" accelerate evaluate rouge-score scikit-learn matplotlib seaborn numpy pandas sentencepiece
```
- Install CUDA-enabled PyTorch per your GPU from the [PyTorch website](https://pytorch.org).
- Recommended: a GPU with ≥8–12 GB VRAM for flan-t5-base; flan-t5-small runs on lower VRAM.

---

## Quick sanity check (zero-shot with FLAN-T5)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Classification (zero-shot)
inp = "sst2: I absolutely loved this movie. It was fantastic!"
out = mdl.generate(**tok(inp, return_tensors="pt"), max_new_tokens=8)
print(tok.decode(out[0], skip_special_tokens=True))  # likely "positive"

# Summarization (zero-shot)
dialogue = "summarize: John: Let's meet at 5 pm.\nJane: Can we do 6 pm instead?\nJohn: Sure. See you then."
out = mdl.generate(**tok(dialogue, return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Training script (FLAN-T5 + PEFT)
Copy the following to `train_peft.py`. It supports Prompt-Tuning, Prefix-Tuning, and LoRA on FLAN-T5.

```python
# name: train_peft.py
import os
import argparse
import numpy as np
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
)
import evaluate
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return mdl, tok

def wrap_with_peft(model, method: str, task_type: TaskType, tokenizer_name: str, capacity: int, target_modules=None):
    if method == "lora":
        # T5 often works well with LoRA on Q and V projections; you can also try ["q","k","v","o"]
        targets = target_modules or ["q", "v"]
        cfg = LoraConfig(
            task_type=task_type,
            r=capacity,                 # capacity knob
            lora_alpha=max(16, 2*capacity),
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",                # T5 layers commonly have no bias
        )
        peft_model = get_peft_model(model, cfg)

    elif method == "prompt":
        cfg = PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=capacity,             # capacity knob
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="You are a helpful assistant.",
            tokenizer_name_or_path=tokenizer_name,
        )
        peft_model = get_peft_model(model, cfg)

    elif method == "prefix":
        cfg = PrefixTuningConfig(
            task_type=task_type,
            num_virtual_tokens=capacity,            # capacity knob
        )
        peft_model = get_peft_model(model, cfg)
    else:
        raise ValueError("method must be one of: lora, prompt, prefix")

    peft_model.print_trainable_parameters()
    return peft_model

def load_and_tokenize(task: str, tok, max_src: int, max_tgt: int):
    if task == "classification":
        ds = load_dataset("glue", "sst2")
        label_map = {0: "negative", 1: "positive"}

        def preprocess(batch):
            inputs = [f"sst2: {s}" for s in batch["sentence"]]
            targets = [label_map[int(l)] for l in batch["label"]]
            enc = tok(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=max_src,
            )
            # modern API: text_target for labels
            with tok.as_target_tokenizer():
                lab = tok(
                    targets,
                    truncation=True,
                    padding="max_length",
                    max_length=max_tgt,
                )
            enc["labels"] = lab["input_ids"]
            return enc

        cols = ds["train"].column_names
        ds_tok = ds.map(preprocess, batched=True, remove_columns=cols)

    elif task == "summarization":
        ds = load_dataset("samsum")

        def preprocess(batch):
            inputs = [f"summarize: {c}" for c in batch["dialogue"]]
            targets = batch["summary"]
            enc = tok(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=max_src,
            )
            with tok.as_target_tokenizer():
                lab = tok(
                    targets,
                    truncation=True,
                    padding="max_length",
                    max_length=max_tgt,
                )
            enc["labels"] = lab["input_ids"]
            return enc

        cols = ds["train"].column_names
        ds_tok = ds.map(preprocess, batched=True, remove_columns=cols)

    else:
        raise ValueError("task must be 'classification' or 'summarization'")

    return ds_tok

def metrics_fn(task: str, tok):
    if task == "classification":
        def normalize_labels(strs: List[str]):
            out = []
            for s in strs:
                s = s.strip().lower()
                if "positive" in s and "negative" in s:
                    out.append("positive" if s.find("positive") <= s.find("negative") else "negative")
                elif "positive" in s:
                    out.append("positive")
                elif "negative" in s:
                    out.append("negative")
                else:
                    # fallback heuristic
                    out.append("positive" if s.startswith("p") else "negative")
            return out

        def compute(eval_pred):
            pred_ids, label_ids = eval_pred
            preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
            # replace -100 before decoding labels (trainer passes -100 for ignored positions)
            label_ids = [[(tid if tid != -100 else tok.pad_token_id) for tid in seq] for seq in label_ids]
            labels = tok.batch_decode(label_ids, skip_special_tokens=True)
            preds = normalize_labels(preds)
            labels = normalize_labels(labels)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, pos_label="positive"),
            }
        return compute

    else:
        rouge = evaluate.load("rouge")
        def compute(eval_pred):
            pred_ids, label_ids = eval_pred
            preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
            label_ids = [[(tid if tid != -100 else tok.pad_token_id) for tid in seq] for seq in label_ids]
            refs = tok.batch_decode(label_ids, skip_special_tokens=True)
            res = rouge.compute(
                predictions=[p.strip() for p in preds],
                references=[r.strip() for r in refs],
                use_stemmer=True
            )
            return {k: round(v * 100, 2) for k, v in res.items()}
        return compute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "summarization"], required=True)
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--method", choices=["lora", "prompt", "prefix"], required=True)
    parser.add_argument("--capacity", type=int, required=True, help="LoRA rank OR number of virtual tokens")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--train_samples", type=int, default=2000, help="subsample for low-resource setting")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # FLAN-T5 + PEFT often tolerates higher LR
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tok = load_model_and_tokenizer(args.model_name)

    peft_model = wrap_with_peft(
        model=model,
        method=args.method,
        task_type=TaskType.SEQ_2_SEQ_LM,
        tokenizer_name=args.model_name,
        capacity=args.capacity,
        target_modules=["q", "v"],  # T5 attention defaults; try ["q","k","v","o"] for a bit more capacity
    )

    if args.task == "classification":
        max_src, max_tgt = 256, 8
    else:
        max_src, max_tgt = 512, 128

    ds_tok = load_and_tokenize(args.task, tok, max_src, max_tgt)
    train_ds = ds_tok["train"]
    if 0 < args.train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.train_samples))
    eval_split = "validation" if "validation" in ds_tok else "test"
    eval_ds = ds_tok[eval_split]

    data_collator = DataCollatorForSeq2Seq(tok, model=peft_model)
    compute_metrics = metrics_fn(args.task, tok)

    run_name = f"{args.task}_{args.method}_cap{args.capacity}_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=max_tgt,
        logging_steps=50,
        report_to=["none"],
        fp16=args.fp16,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if args.task == "classification" else "rougeL",
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    # Count trainable vs total params
    try:
        trainable, total = peft_model.get_nb_trainable_parameters()
    except:
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable params: {trainable} | Total params: {total} | Ratio: {trainable/total:.6f}")

if __name__ == "__main__":
    main()
```

Example runs (FLAN-T5):
```bash
# Classification with LoRA (rank 8)
python train_peft.py --task classification --method lora --capacity 8 --fp16

# Classification with Prompt-Tuning (20 virtual tokens)
python train_peft.py --task classification --method prompt --capacity 20 --fp16

# Summarization with Prefix-Tuning (50 virtual tokens)
python train_peft.py --task summarization --method prefix --capacity 50 --fp16
```

Tips specific to FLAN-T5:
- Start with google/flan-t5-small; once stable, try flan-t5-base if VRAM allows.
- T5 often benefits from LoRA on ["q","v"]; adding ["k","o"] increases capacity and compute slightly.
- Keep LR in 1e-4 to 5e-4 for LoRA/Prefix; Prompt-Tuning sometimes likes higher LR (e.g., 1e-3 to 5e-3). The default 5e-4 is a safe start.

---

## Aggregation and plotting
Use this helper to aggregate results from `outputs/` and create the curves. Save as `plot_curves.py`.

```python
# name: plot_curves.py
import json, os, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_cfg(run_name):
    # {task}_{method}_cap{capacity}_seed{seed}
    m = re.match(r"(classification|summarization)_(lora|prompt|prefix)_cap(\d+)_seed(\d+)", run_name)
    if not m: return None
    return dict(task=m.group(1), method=m.group(2), capacity=int(m.group(3)), seed=int(m.group(4)))

def load_metrics(run_dir, task):
    # Prefer all_results.json; fall back to trainer_state.json
    ar = os.path.join(run_dir, "all_results.json")
    if os.path.exists(ar):
        with open(ar) as f:
            res = json.load(f)
        if task == "classification":
            return res.get("eval_f1", res.get("eval_accuracy"))
        else:
            # prefer ROUGE-L
            return res.get("eval_rougeL", res.get("eval_rouge1"))
    ts = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(ts):
        with open(ts) as f:
            st = json.load(f)
        # st.get("best_metric") may be scalar; use as-is
        return st.get("best_metric")
    return None

def main(outputs_dir="outputs", out_csv="results.csv"):
    rows = []
    for run in os.listdir(outputs_dir):
        cfg = parse_cfg(run)
        if not cfg: continue
        metric = load_metrics(os.path.join(outputs_dir, run), cfg["task"])
        if metric is None: continue
        rows.append({**cfg, "metric": metric})
    if not rows:
        print("No results found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)

    for task in df.task.unique():
        d = df[df.task == task]
        plt.figure(figsize=(6,4))
        sns.lineplot(data=d, x="capacity", y="metric", hue="method", marker="o")
        plt.xscale("log", base=2)
        plt.title(f"{task.capitalize()} (FLAN-T5): Parameter Efficiency")
        plt.xlabel("Capacity (LoRA rank r or num_virtual_tokens)")
        plt.ylabel("F1" if task=="classification" else "ROUGE-L")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"curve_{task}.png", dpi=200)
        plt.show()

if __name__ == "__main__":
    main()
```

Run:
```bash
# Sweep capacities for each method and task
for r in 4 8 16 32; do
  python train_peft.py --task classification --method lora --capacity $r --fp16
done
for n in 10 20 50 100; do
  python train_peft.py --task classification --method prompt --capacity $n --fp16
  python train_peft.py --task classification --method prefix --capacity $n --fp16
done

for r in 4 8 16 32; do
  python train_peft.py --task summarization --method lora --capacity $r --fp16
done
for n in 10 20 50 100; do
  python train_peft.py --task summarization --method prompt --capacity $n --fp16
  python train_peft.py --task summarization --method prefix --capacity $n --fp16
done

python plot_curves.py
```

Outputs:
- `results.csv` with rows: task, method, capacity, seed, metric
- `curve_classification.png`, `curve_summarization.png`

---

## Suggested experiment grid (FLAN-T5)
- Model: google/flan-t5-small (then base if possible)
- Data size: `--train_samples 2000` to simulate low-resource
- Fixed training budget: 3 epochs, LR=5e-4, same batch sizes
- Capacity sweeps:
  - LoRA r ∈ {4, 8, 16, 32}
  - Prompt/Prefix num_virtual_tokens ∈ {10, 20, 50, 100}
- Optional: 2–3 seeds per config and average

---

## Expected trends on FLAN-T5
- LoRA scales well with r; r≈16 is a strong sweet spot on small/base.
- Prompt/Prefix are very parameter-efficient; gains plateau beyond ~50–100 tokens.
- Summarization generally needs more capacity than classification.
- Zero-shot FLAN-T5 is a decent baseline; PEFT can surpass it with moderate capacity.

---

## Reproducibility and tips
- Set seeds: `--seed 42`
- VRAM tips:
  - flan-t5-small: fits on modest GPUs; try batch size 16 with fp16
  - flan-t5-base: may need smaller batch size or gradient accumulation
- If OOM:
  - Reduce `--per_device_train_batch_size`
  - Shorten `max_source_len` / `max_target_len`
  - Use flan-t5-small
- If training stalls:
  - Try LR in [1e-4, 1e-3]
  - Increase capacity slightly
  - Train a few more epochs with early stopping
- Fairness: keep preprocessing, prompts, and training budget identical across methods
- Count parameters and log everything per run

---

## References
- FLAN-T5: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Prefix-Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- Libraries:
  - [Hugging Face Transformers](https://github.com/huggingface/transformers)
  - [PEFT](https://github.com/huggingface/peft)
  - [Datasets](https://github.com/huggingface/datasets)
  - [Evaluate](https://github.com/huggingface/evaluate)

---

## License
Educational use only. Check dataset and model licenses before redistribution.