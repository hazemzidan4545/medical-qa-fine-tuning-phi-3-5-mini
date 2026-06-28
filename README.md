# Medical Q&A Fine-Tuning with Phi-3.5-mini

This project fine-tunes `microsoft/Phi-3.5-mini-instruct` for medical question answering using the Medical Meadow Medical Flashcards dataset.

The notebook compares two parameter-efficient fine-tuning approaches:

- LoRA adapters
- Prompt tuning adapters

## Contents

- `NLP.ipynb` - main experiment notebook for preprocessing, training, and evaluation.
- `NLP CW1 Report.docx` - coursework report.
- `comparison_chart.png` - result comparison chart.
- `lora_adapter/` - exported LoRA adapter files.
- `prompt_tuning_adapter/` - exported prompt tuning adapter files.

Generated checkpoint folders are ignored by Git because they can be recreated from the notebook and make the repository much larger.

## Base Model

- `microsoft/Phi-3.5-mini-instruct`

## Dataset

- `medalpaca/medical_meadow_medical_flashcards`
