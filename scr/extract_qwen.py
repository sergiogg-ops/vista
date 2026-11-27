import json
import os

from torch.utils.data import Dataset
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from configargparse import ArgumentParser
from transformers import AutoProcessor
from scr.baselines.train_qwen import QwenOCRVLTranslator, CODE2LANG
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, if present

def parse_args():
    parser = ArgumentParser(description="Fine-tune GemmaOCR-VL-1.6 for image -> English translation.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", is_config_file=True)

    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the images to translate.")
    parser.add_argument("--target_lang", type=str, required=True, help="Language to which translate the images.")
    parser.add_argument("--model_path", type=str, default="GemmaGemma/GemmaOCR-VL-1.6", help="HF repo id or local path to GemmaOCR-VL-1.6.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HF auth token, if the model is private.")

    parser.add_argument("--prompt", type=str, default="Translate the text in this image into English:", help="Instruction prompt used for every sample.")
    parser.add_argument("--image_token", type=str, default=None, help="Literal image placeholder token used by the model's chat template. Defaults to processor.image_token if available.")
    parser.add_argument("--merge_size", type=int, default=2, help="Spatial merge size of the vision encoder (used to turn patch grid dims into placeholder-token count). Verify against your model's config.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total (prompt + target) token length; sequences are truncated to this.")

    parser.add_argument("--output_file", type=str, required=True, help="Where to write the translations.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Use the model repo's custom Hub code instead of transformers' native GemmaOCR-VL implementation. Only needed on older transformers versions without native support; leave off on transformers>=5.x, which now ships GemmaOCRVLForConditionalGeneration natively.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    #os.makedirs(args.output_file, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, token=args.hf_token
    )

    kwargs= {
        'model_path': args.model_path,
        'tokenizer':processor.tokenizer,
        'attn_implementation':args.attn_implementation,
        'token':args.hf_token,
        'trust_remote_code':args.trust_remote_code,
        'use_lora':True,
        'lora_r': 8,
        'lora_alpha': 16,
    }
    model = QwenOCRVLTranslator.load_from_checkpoint(args.checkpoint, map_location='cpu', weights_only=False, **kwargs)

    model.model.save_pretrained(args.output_file)

if __name__ == "__main__":
    main()