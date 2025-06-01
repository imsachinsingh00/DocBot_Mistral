
# 🧠 LoRA Fine-Tuning for Mistral-7B on MTS-Dialog Dataset

This project demonstrates parameter-efficient fine-tuning (PEFT) using LoRA on the [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) language model to generate concise section summaries from doctor-patient dialogues in the [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) dataset.

---

## 📁 Project Structure

```
LoRA_Mistral_MTSDialog_Summarization/
├── train_finetune.ipynb       # Main training and evaluation notebook
├── data/                      # CSV source for MTS-Dialog dataset (if local)
├── mistral-mts-summary_1/     # Output LoRA adapter and tokenizer
└── README.md                  # You're here
```

---

## 🚀 Approach

- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Quantization**: 4-bit (using `bitsandbytes`)
- **PEFT Method**: Low-Rank Adaptation (LoRA)
- **LoRA Config**: `rank=4`, `alpha=32`, `target_modules=["q_proj", "v_proj"]`
- **Tokenizer Adjustments**: Added `[PAD]` token if not present
- **Training**:
  - Epochs: `3`
  - Batch Size: `4` (gradient accumulation used for larger effective batch)
  - Evaluation: Performed at the end of each epoch
  - Device: `cuda:1` (customizable)

---

## 📊 Evaluation

The model was evaluated on the validation split using standard summarization metrics:

```
Final validation metrics:
- ROUGE-1: **0.1318**
- ROUGE-2: **0.0456**
- ROUGE-L: **0.0900**
- BLEU:    **0.0260**
```

### ⚠️ Notes on Low Performance

These scores are relatively low for a summarization task, and this is expected due to:

- 🔧 **Limited training budget**: Only **3 epochs** of training were performed.
- 🧠 **Low-rank adaptation**: LoRA was applied with a **rank of just 4**, which restricts the model’s capacity to adapt.
- 🚫 **Constrained compute**: Training was performed on a system with limited GPU memory, restricting batch size and training duration.

Despite the modest scores, this prototype demonstrates a complete LoRA-based fine-tuning pipeline using `mistralai/Mistral-7B-v0.1`, suitable for low-resource environments and further extension.

---

## 🛠️ Dependencies

Make sure the following libraries are installed:

```bash
pip install transformers datasets evaluate peft bitsandbytes accelerate
```

---

## 📌 Notes

- The `load_in_4bit` argument is now deprecated. For future use, replace it with:
```python
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_4bit=True, ...)
AutoModelForCausalLM.from_pretrained(..., quantization_config=quant_config)
```
- Token embedding layers were resized with a multivariate normal distribution to accommodate `[PAD]` token.

---

## 📬 Contact

Maintained by [Sachin Singh](https://github.com/imsachinsingh00)
