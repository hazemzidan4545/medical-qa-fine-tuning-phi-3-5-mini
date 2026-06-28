---
base_model: microsoft/Phi-3.5-mini-instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:microsoft/Phi-3.5-mini-instruct
- prompt-tuning
- sft
- transformers
- trl
---

# Prompt Tuning Adapter for Medical Q&A

This directory contains the exported prompt tuning adapter trained for medical question answering with `microsoft/Phi-3.5-mini-instruct`.

## Model Details

- Base model: `microsoft/Phi-3.5-mini-instruct`
- Dataset: `medalpaca/medical_meadow_medical_flashcards`
- Task type: causal language modeling
- PEFT method: prompt tuning
- PEFT version: 0.18.1

## Training Setup

- Train split: 2,000 examples
- Evaluation split: 200 examples
- Test split: 100 examples
- Maximum sequence length: 512 tokens
- Quantization: 4-bit BitsAndBytes
- Optimizer: `paged_adamw_8bit`
- Learning rate: `1e-3`
- Batch size per device: 2
- Gradient accumulation steps: 8
- Matched training steps: 750
- Scheduler: constant with warmup

## Prompt Tuning Configuration

- Virtual tokens: 20
- Initialization: text
- Initialization text: `Answer the following medical question accurately:`
- Token dimension: 3072
- Attention heads: 32
- Layers: 32
- Trainable parameters: 61,440
- Trainable percentage: 0.0016%

## Evaluation

The adapter was evaluated on 50 test examples from the notebook split.

| Model | ROUGE-L | BERTScore F1 | Perplexity |
| --- | ---: | ---: | ---: |
| Baseline (Pure Base Model) | 0.1991 | 0.8624 | 11.67 |
| Prompt Tuning Fine-Tuned | 0.2485 | 0.8762 | 2.09 |

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "./prompt_tuning_adapter")
```

## Framework Versions

- PEFT 0.18.1
