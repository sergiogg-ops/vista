# VISTA: Visual-Integrated Space Translation Adapter

Official code for **"End-to-End Image-to-Text Machine Translation via Visual-Integrated Space Translation Adapter"** (WMT 2026).

VISTA translates the text embedded in an image directly into a target language, without an intermediate OCR step. It connects a pretrained vision encoder (PaddleOCR-VL-1.6) to a pretrained MT decoder (NLLB-200) through a Perceiver Resampler trained from scratch. For the method, experiments and results, please refer to the paper; this README only explains how to use the code and reproduce the experiments.

```
image → vision encoder → adapter (Perceiver Resampler) → NLLB decoder → translation
                                  ├─ L_ctc   (auxiliary CTC loss on the transcription)
                                  └─ L_align (alignment with the NLLB encoder space)
```

## Repository structure

```
vista/
├── config/                    # YAML configs (precompute, training, inference)
├── scr/
│   ├── vista.py               # VISTA model (resampler + Translator LightningModule) and datasets
│   ├── train_vista.py         # Train VISTA
│   ├── inf_vista.py           # Inference with VISTA
│   ├── translate.py           # Text-to-text MT, used to extend the datasets to new languages
│   ├── precompute/{OCRMT30K,LSVT}/   # precompute_paddle.py (VISTA) · precompute_images.py (VLM baselines)
│   ├── baselines/             # Fine-tuning of the VLM baselines
│   └── inf_{qwen,gemma,deepseek,paddle}.py   # Inference of the VLM baselines
├── tester/                    # Scripts that run inference over all target languages
├── create_test_launcher.sh    # Commands used to translate the datasets into the extra languages
├── .env.example
└── LICENSE
```

## Getting started

### Installation

Requirements: Linux, a CUDA GPU (training and inference scripts require one) and [conda](https://docs.conda.io/). The pinned dependencies in `requirements.txt` correspond to the environment used for the experiments (PyTorch 2.14, Transformers 5.18, PyTorch Lightning 2.6; CUDA 13 builds of the PyTorch libraries, so a recent NVIDIA driver is needed).

```bash
git clone https://github.com/sergiogg-ops/vista.git
cd vista

conda create -n vista python=3.14
conda activate vista

# 1. Install PyTorch for YOUR GPU / CUDA driver.
#    Pick the command at https://pytorch.org/get-started/locally/, e.g.:
pip install torch

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

Two optional packages are not part of `requirements.txt`:

- [`ollama`](https://pypi.org/project/ollama/) (and a running Ollama server), imported by `scr/translate.py`.
- `torchvision`, imported by `scr/baselines/train_deepseek.py`.

```bash
pip install ollama torchvision
```

A recent `transformers` release is required, since the Qwen3.5, Gemma 4, PaddleOCR-VL-1.6 and DeepSeek-OCR-2 models are used.

### Credentials

```bash
cp .env.example .env     # then fill in HF_TOKEN and WANDB_API_KEY
```

`HF_TOKEN` is used to download models from the Hugging Face Hub and `WANDB_API_KEY` for experiment tracking.

### Data

The data is not included in this repository. Place it in the repository root (these folders are git-ignored, as are `hypotheses/` and `checkpoints/`):

- **OCRMT30K**: images from [`zhibinlan/OCRMT30K`](https://huggingface.co/datasets/zhibinlan/OCRMT30K) (Lan et al., 2023) → `OCRMT30k/`, using the `before_tok_bpe/` (index, transcription and translation files), `raw_data/` (bounding boxes) and image folders.
- **ICDAR2019-LSVT**: [LSVT](https://rrc.cvc.uab.es/?ch=16) images and annotations → `LSVT/`. As the official test references are not public, the paper uses 1,000 images randomly held out from the training set.

Expected structure:

```
OCRMT30k/  before_tok_bpe/  raw_data/  valid_images/  ...
LSVT/      test_zh.txt  test_zh.json  test_full_images/  test_<lang>_nllb.txt
```

#### Extending the datasets to more languages

The paper extends both datasets with nine additional target languages (Arabic, Bengali, Spanish, French, Hindi, Indonesian, Japanese, Russian, Urdu) by machine-translating the Chinese transcriptions. This is done with `scr/translate.py`:

```bash
python scr/translate.py <source_file> <output_file> \
    --src_lang zho --src_code Hans --tgt_lang spa --tgt_code Latn \
    --model facebook/nllb-200-3.3B --batch_size 2
```

- `facebook/nllb-200-3.3B` produces the translations used for training and as first reference.
- `--model translategemma:27b` (through [Ollama](https://ollama.com/library/translategemma), with ISO codes such as `--src_lang Chinese --src_code zh`) produces the second reference of the test sets.

`create_test_launcher.sh` lists the exact commands for every language, dataset and model.

### Preprocessing of OCRMT30k and LSVT

Both datasets are preprocessed in the same way: each image is cropped to the bounding box of its text, and the transcriptions and translations are tokenized with the NLLB tokenizer. Run the scripts once per split (train / valid / test), changing the paths and `output_dir` in the config (e.g. `OCRMT30k/precomputed_train`, `OCRMT30k/precomputed_val`, `OCRMT30k/precomputed_test`).

**For VISTA**: stores `image.safetensors` and the token ids of the source and each target language for every sample, plus `texts_<lang>.json`.

```bash
python scr/precompute/OCRMT30K/precompute_paddle.py --config config/precompute_ocrmt30k.yaml
python scr/precompute/LSVT/precompute_paddle.py     --config config/precompute_lsvt.yaml
```

**For the VLM baselines**: stores the cropped `.jpg` images plus `texts_<lang>.json`.

```bash
python scr/precompute/OCRMT30K/precompute_images.py --config config/precompute_ocrmt30k.yaml
python scr/precompute/LSVT/precompute_images.py     --config config/precompute_lsvt.yaml
```

> `output_dir` is **emptied** before writing. `config/precompute_ocrmt30k.yaml` is currently set for the validation split.

## Training a VISTA model

```bash
python scr/train_vista.py --config config/train_vista.yaml
```

Edit `train_latents_dir` and `valid_latents_dir` in the config to point to your own precomputed data (they currently hold absolute paths of the authors' machine). Any option can also be passed on the command line and overrides the config.

The paper's setting (batch of 128 samples, LoRA with r = 8 and α = 16, AdamW, cosine annealing from 1e-4 to 1e-6, λ_mt = 1, λ_ctc = 0.05, λ_align = 1) maps to these fields:

| Paper | Config field |
|-------|--------------|
| Batch of 128 samples | `train_batch_size: 4` × `accumulate_grad: 32` |
| Loss coefficients | `mt_weight`, `ctc_weight`, `align_weight` |
| Staged unfreezing: top 2 vision layers → decoder cross-attention → top 2 decoder layers | `unfreeze_milestones` (optimizer steps, in that order) |
| Learning rates | `ini_lr` (adapter), `lr_vision`, `lr_decoder`, `min_lr` |
| LoRA on the vision encoder | `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout` |
| Early stopping (patience 3) | `early_stopping`, `patience` |
| Adapter (4 layers, 16 heads) | `adapter_args` |


Checkpoints (top 3 by validation BLEU) are saved in `checkpoints/` as `<name>-step=<N>-val_BLEU=<score>.ckpt`, and metrics are logged to Weights & Biases (`project`, `name`).

## Inference of VISTA models

Set the checkpoint, the precomputed test data (`latents_dir`) and the source language in `config/inference_vista.yaml`. The adapter and LoRA settings must match those used in training. Each run translates into one target language:

```bash
python scr/inf_vista.py \
    --config config/inference_vista.yaml \
    --tgt_lang eng_Latn \
    --output_file hypotheses/lsvt_zh-en/lsvt_test_en_vista.json
```

The output is a JSON list of `{"idx": ..., "text": ...}` records. To translate into all ten target languages:

```bash
bash tester/vista.sh
```

# Baselines

### VLM baselines

The paper compares VISTA with four open-source VLMs, used zero-shot and fine-tuned on OCRMT30k (Qwen3.5 could only be evaluated zero-shot):

| Model | Fine-tuning | Inference | Config | All-languages script |
|-------|-------------|-----------|--------|----------------------|
| Gemma-4-E2B-it | `scr/baselines/train_gemma.py` | `scr/inf_gemma.py` | `config/inference_gemma.yaml` | `tester/gemma.sh` |
| DeepSeek-OCR-2 | `scr/baselines/train_deepseek.py` | `scr/inf_deepseek.py` | `config/inference_deepseek.yaml` | `tester/deepseek.sh` |
| Qwen3.5-4B | `scr/baselines/train_qwen.py` | `scr/inf_qwen.py` | `config/inference_qwen.yaml` | `tester/qwen.sh` |
| PaddleOCR-VL-1.6 | `scr/baselines/train_paddle.py` | `scr/inf_paddle.py` | `config/inference_paddle.yaml` | `tester/paddleocr.sh` |

**Fine-tuning.** The baselines use the cropped images from `precompute_images.py`. `config/train_baseline.yaml` holds the shared settings (LoRA, frozen vision encoder, effective batch of 128, learning rate 1e-5, early stopping on `val_loss`) and the prompt of each model. Uncomment the `model_path`/`prompt` block of the model to train (Qwen is active by default), set `data_dir` and `val_data_dir`, and run the matching script:

```bash
python scr/baselines/train_gemma.py --config config/train_baseline.yaml
```

**Inference.**

```bash
python scr/inf_gemma.py \
    --config config/inference_gemma.yaml \
    --target_lang eng_Latn \
    --output_file hypotheses/lsvt_zh-en/lsvt_test_en_gemma.json
```

`data_dir` is the folder with the cropped test images, and `prompt` uses the `{source_lang}` / `{target_lang}` placeholders (see the paper's appendix for the prompts). To evaluate a fine-tuned model, set `resume_from_checkpoint` to its checkpoint, as in `config/inference_paddle.yaml`.

### Cascade baseline

The cascade baseline of the paper (EasyOCR transcription followed by NLLB translation, no fine-tuning) is **not** part of this repository. It can be reproduced using [this repo](https://github.com/sergiogg-ops/modular_i2imt).

## Evaluation

We have performed the evaluation using the [DEEP software](https://github.com/sergiogg-ops/deep).

## Citation

The paper is currently under review. The citation will be added here once it is published.

## License

Released under the [MIT License](LICENSE).