---
base_model: microsoft/Phi-3.5-mini-instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:microsoft/Phi-3.5-mini-instruct
- lora
- sft
- transformers
- trl
---

# LoRA Adapter for Medical Q&A

This directory contains the exported LoRA adapter trained for medical question answering with `microsoft/Phi-3.5-mini-instruct`.

## Model Details

- Base model: `microsoft/Phi-3.5-mini-instruct`
- Dataset: `medalpaca/medical_meadow_medical_flashcards`
- Task type: causal language modeling
- PEFT method: LoRA
- PEFT version: 0.18.1

## Training Setup

- Train split: 2,000 examples
- Evaluation split: 200 examples
- Test split: 100 examples
- Maximum sequence length: 512 tokens
- Quantization: 4-bit BitsAndBytes
- Optimizer: `paged_adamw_8bit`
- Learning rate: `2e-4`
- Epochs: 3
- Batch size per device: 2
- Gradient accumulation steps: 4
- Scheduler: cosine

## LoRA Configuration

- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Bias: none
- Trainable parameters: 3,145,728
- Trainable percentage: 0.0823%

## Evaluation

The adapter was evaluated on 50 test examples from the notebook split.

| Model | ROUGE-L | BERTScore F1 | Perplexity |
| --- | ---: | ---: | ---: |
| Baseline (Pure Base Model) | 0.1991 | 0.8624 | 11.67 |
| LoRA Fine-Tuned | 0.1995 | 0.8629 | 11.25 |

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "./lora_adapter")
```

## Framework Versions

- PEFT 0.18.1
