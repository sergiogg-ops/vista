# interlingua_img

Image-to-text machine translation for multilingual OCR and visual text understanding.

This repository implements an image-to-text machine translation pipeline that reads text embedded in images, extracts visual features with a vision encoder, and translates the recognized content into another language using a multilingual sequence-to-sequence model. The project is designed for research and experimentation in document understanding, OCR-based translation, and multimodal machine translation.

## Overview

The core idea is to translate text from images rather than only translating plain text strings. The system combines:

- a visual encoder for image understanding
- a multilingual translation model (NLLB-based in the main workflow)
- adapter modules that bridge visual representations into the text generation model
- preprocessing and inference scripts for multilingual OCR and translation benchmarks

This makes it useful for real-world tasks such as translating signs, documents, street text, product labels, and screenshots across languages.

## What the project does

The repository trains and runs models that:

- ingest image inputs containing text
- extract visual features and latent representations
- translate the represented text into one or more target languages
- evaluate translation quality on multilingual OCR datasets such as OCRMT30K and LSVT

The main focus is image-to-text machine translation, often in the setting of multilingual visual text translation.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── .env.example
├── train.py
├── create_test_launcher.sh
├── config/
│   ├── train.yaml
│   ├── train_baseline.yaml
│   ├── inference_deepseek.yaml
│   ├── inference_gemma.yaml
│   ├── inference_paddle.yaml
│   ├── inference_qwen.yaml
│   ├── inference_vista.yaml
│   ├── precompute_lsvt.yaml
│   └── precompute_ocrmt30k.yaml
├── scr/
│   ├── translate.py
│   ├── translate.ipynb
│   ├── inf_deepseek.py
│   ├── inf_gemma.py
│   ├── inf_paddle.py
│   ├── inf_qwen.py
│   ├── inf_vista.py
│   ├── precompute/
│   │   ├── LSVT/
│   │   └── OCRMT30K/
│   ├── baselines/
│   │   ├── deeseek.py
│   │   ├── gemma.py
│   │   ├── paddle.py
│   │   └── qwen.py
│   └── vista/
│       ├── mt_clip.py
│       ├── mt_deepseek.py
│       ├── mt_paddle.py
│       ├── mt_sam.py
│       ├── mt_siglip.py
│       └── mt_trocr.py
├── tester/
│   ├── deepseek.sh
│   ├── evaluate.sh
│   ├── gemma.sh
│   ├── paddleocr.sh
│   ├── qwen.sh
│   └── vista.sh
└── ...
```

## Training pipeline

The main training script is `train.py`. It loads configuration from YAML files, initializes a multilingual translation model and a vision encoder, and trains an image-conditioned translator on precomputed latent features.

The repository is configured for multilingual image-text translation using a setup like:

- source language: `zho_Hans` (Chinese)
- target languages: English, Arabic, Bengali, Spanish, French, Hindi, Indonesian, Japanese, Russian, and Urdu
- vision model: `PaddlePaddle/PaddleOCR-VL-1.6`
- machine translation model: `facebook/nllb-200-distilled-600M`
- LoRA tuning enabled
- early stopping and checkpointing enabled

Main configuration lives in `config/train.yaml` and can be launched with:

```bash
python train.py --config config/train.yaml
```

The training workflow uses Hugging Face Transformers, PyTorch Lightning, PEFT, Weights & Biases, and dotenv-based configuration.

## Inference pipeline

The repository includes inference scripts for multiple model backbones:

- `scr/inf_vista.py`
- `scr/inf_paddle.py`
- `scr/inf_gemma.py`
- `scr/inf_qwen.py`
- `scr/inf_deepseek.py`

These scripts load a checkpoint or pretrained model and generate translated text from visual-image data.

Example:

```bash
python scr/inf_vista.py \
  --config config/inference_vista.yaml \
  --output_file hypotheses/lsvt_zh-en/lsvt_test_en_vista.json \
  --tgt_lang eng_Latn
```

The shell scripts in `tester/` run the model over multiple languages and output JSON translation files for evaluation.

## Text translation helper

The repository also contains a generic translation utility in `scr/translate.py`. This script can translate text files or JSON datasets using either NLLB or Gemma-based translators.

Example usage:

```bash
python scr/translate.py \
  LSVT/test_zh.txt \
  LSVT/test_en_gemma.txt \
  --src_lang Chinese --src_code zh \
  --tgt_lang English --tgt_code en \
  --model translategemma:27b \
  --batch_size 8
```

This utility is useful for preparing multilingual translated references and for baseline comparisons.

## Precomputation and data flow

The project includes scripts to precompute image features and latent representations before training or inference:

- `scr/precompute/LSVT/precompute_images.py`
- `scr/precompute/LSVT/precompute_paddle.py`
- `scr/precompute/OCRMT30K/precompute_images.py`
- `scr/precompute/OCRMT30K/precompute_paddle.py`
- `scr/precompute/OCRMT30K/precompute_gemma3.py`

Precomputing latents helps accelerate iterative training and supports repeated experimentation with different translation heads and adapters.

## Datasets and evaluation

The project is designed around multilingual text-in-image benchmarks, especially:

- OCRMT30K
- LSVT

The config files reference dataset directories such as:

- `OCRMT30k/precomputed_train`
- `OCRMT30k/precomputed_val`
- `OCRMT30k/precomputed_test`
- `LSVT/test_zh.txt`

The repository also includes evaluation-related scripts such as `tester/evaluate.sh`.

## Environment setup

Create a local environment file from the example:

```bash
cp .env.example .env
```

Then set your values in `.env`:

```dotenv
HF_TOKEN=your huggingface token here
WANDB_API_KEY=your wandb api key here
```

The project uses Hugging Face models and can log training results to Weights & Biases.

## Dependencies

Typical dependencies include:

- Python 3.10+
- PyTorch
- PyTorch Lightning
- Transformers
- PEFT
- configargparse
- PyYAML
- python-dotenv
- tqdm
- Ollama (for Gemma-based translation experiments)

## Example workflow

1. Precompute image features or load existing latents.
2. Configure the run in `config/train.yaml`.
3. Train the model:

```bash
python train.py --config config/train.yaml
```

4. Run inference with a checkpoint:

```bash
python scr/inf_vista.py --config config/inference_vista.yaml --checkpoint checkpoints/<model>.ckpt --output_file hypotheses/output.json
```

5. Evaluate the generated translations using the repository's evaluation scripts.

## Notes

- This project focuses on image-to-text machine translation and multimodal multilingual generation.
- It supports several vision encoders and translation backbones for experimentation.
- It is oriented toward research, benchmarking, and OCR-driven multilingual translation tasks.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This project builds on pretrained multilingual and multimodal models from Hugging Face, including NLLB and PaddleOCR-VL style vision-language systems.
