import json
import os

from torch.utils.data import Dataset
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from configargparse import ArgumentParser
from transformers import AutoProcessor
from scr.baselines.train_gemma import GemmaOCRVLTranslator, CODE2LANG
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, if present

class GemmaOCRTranslationDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        processor, 
        prompt: str = "Translate the {source_lang} text in this image into {target_lang}:", 
        target: str = "eng_Latn"
    ):
        self.records = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg')]
        self.processor = processor
        self.prompt = prompt
        self.tgt_lang = target

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        # Pass only raw data; let the collator handle batch-level processing
        record = self.records[idx]
        image = Image.open(record).convert("RGB")
        
        return {
            "image": image,
            "lang": self.tgt_lang,
            "name": os.path.basename(record).split('.')[0]
        }
    
    def _collate_fn(self, batch_samples):
        images = [s["image"] for s in batch_samples]
        
        # 1. Build string sequences using the model's chat template
        prompt_texts = []
        full_texts = []
        
        for s in batch_samples:
            user_msg = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt.format(
                        source_lang="Chinese",
                        target_lang=CODE2LANG[s["lang"]]
                    )},
                ],
            }]
            
            # Text layout for validation/generation evaluation
            p_text = self.processor.apply_chat_template(
                user_msg, add_generation_prompt=True, tokenize=False
            )
            prompt_texts.append(p_text)

        # 4. Process Validation/Generation Evaluation Batch (Left Padded)
        # Left-padding is mandatory for .generate() so the model steps sequentially from the edge
        self.processor.tokenizer.padding_side = "left"
        val_inputs = self.processor(
            text=prompt_texts, 
            images=images, 
            return_tensors="pt", 
            padding="longest"
        )

        # 5. Assemble final batch output dictionary
        collated_batch = {
            "prompt_input_ids": val_inputs["input_ids"],
            "prompt_attention_mask": val_inputs["attention_mask"],
            "pixel_values": val_inputs["pixel_values"],
            "name": [s["name"] for s in batch_samples]
        }
                
        for k, v in val_inputs.items():
            if k not in ["input_ids", "attention_mask", "pixel_values"]:
                collated_batch[f"prompt_{k}"] = v

        return collated_batch

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

    dataset = GemmaOCRTranslationDataset(
        args.data_dir,
        processor=processor,
        prompt=args.prompt,
        target=args.target_lang
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset._collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
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
    if args.resume_from_checkpoint:
        model = GemmaOCRVLTranslator.load_from_checkpoint(args.resume_from_checkpoint, map_location='cpu', weights_only=False, **kwargs)
        model = model.to('cuda')
    else:
        model = GemmaOCRVLTranslator(**kwargs)


    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        enable_progress_bar=True,
        logger=False,
    )

    hypotheses = trainer.predict(model, dataloaders=dataloader)
    
    records = [
        {"idx": name,
            "text": hyp} for batch in hypotheses for name, hyp in zip(batch[0], batch[1])
    ]

    with open(args.output_file, 'w') as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    print(f"{len(records)} translations saved to {args.output_file}.")

if __name__ == "__main__":
    main()
