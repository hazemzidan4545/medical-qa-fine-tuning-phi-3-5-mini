# Medical Q&A Fine-Tuning with Phi-3.5-mini

This project fine-tunes `microsoft/Phi-3.5-mini-instruct` for medical question answering using the `medalpaca/medical_meadow_medical_flashcards` dataset.

The notebook compares two parameter-efficient fine-tuning methods:

- LoRA
- Prompt tuning

## Repository Contents

- `NLP.ipynb` - experiment notebook for setup, data processing, training, evaluation, and comparison.
- `comparison_chart.png` - saved chart from the final metrics comparison.
- `lora_adapter/` - exported LoRA adapter.
- `prompt_tuning_adapter/` - exported prompt tuning adapter.

## Experiment Setup

- Base model: `microsoft/Phi-3.5-mini-instruct`
- Dataset: `medalpaca/medical_meadow_medical_flashcards`
- Maximum sequence length: 512 tokens
- Train split: 2,000 examples
- Evaluation split: 200 examples
- Test split: 100 examples
- Evaluation sample size: 50 test examples
- Metrics: ROUGE-L, BERTScore F1, and perplexity

The notebook formats each example as a Phi-3.5 chat prompt with system, user, and assistant turns. It removes short answers, filters overly long examples, deduplicates by question, and evaluates the base model before comparing both fine-tuned adapters.

## Fine-Tuning Methods

### LoRA

- PEFT type: `LORA`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Trainable parameters: 3,145,728
- Trainable percentage: 0.0823%

### Prompt Tuning

- PEFT type: `PROMPT_TUNING`
- Virtual tokens: 20
- Initialization text: `Answer the following medical question accurately:`
- Trainable parameters: 61,440
- Trainable percentage: 0.0016%
- Matched training steps: 750

## Results

| Model | ROUGE-L | BERTScore F1 | Perplexity |
| --- | ---: | ---: | ---: |
| Baseline (Pure Base Model) | 0.1991 | 0.8624 | 11.67 |
| LoRA Fine-Tuned | 0.1995 | 0.8629 | 11.25 |
| Prompt Tuning Fine-Tuned | 0.2485 | 0.8762 | 2.09 |

The notebook summary reports that fine-tuning improved medical QA behavior over the base model in this run. It also notes that LoRA showed the strongest qualitative balance across sampled answers, while prompt tuning achieved the best aggregate metrics in the final table.

## Dependencies

The notebook installs and uses:

- `transformers==4.46.1`
- `datasets`
- `peft`
- `bitsandbytes`
- `accelerate`
- `trl`
- `rouge-score`
- `bert-score`
- `tensorboard`
