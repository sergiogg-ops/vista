import json
import os

from torch.utils.data import Dataset
from PIL import Image
import pytorch_lightning as pl
import torch
from configargparse import ArgumentParser
from transformers import  AutoTokenizer
from scr.baselines.train_deepseek import preprocess_prompt_with_image, DeepSeekOCRTranslator, CODE2LANG
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, if present

class DeepSeekOCRTranslationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        prompt: str = "<image>\n<|grounding|>Translate the text in this image into {lang}. ",
        target: str = "eng_Latn",
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
        image_token: str = "<image>",
        image_token_id: int = 128815,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        crop_tile_size: int = 768,
        max_length: int = 512,
    ):
        self.records = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg')]
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tgt_lang = target
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.crop_tile_size = crop_tile_size
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record).convert("RGB")
        lang = self.tgt_lang

        return {
            "image": image,
            "lang": lang,
            "prompt_text": self.prompt.format(lang=CODE2LANG[lang]),
            'name': os.path.basename(record).split('.')[0]
        }

    def _collate_fn(self, batch_samples):
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id  # DeepSeek tokenizers often don't define a pad token

        full_ids_list, full_mask_list = [], []
        prompt_ids_list, prompt_mask_list = [], []
        images_list, spatial_crop_list = [], []

        for s in batch_samples:
            pre = preprocess_prompt_with_image(
                tokenizer=self.tokenizer,
                image=s["image"],
                prompt_text=s["prompt_text"],
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                image_token=self.image_token,
                image_token_id=self.image_token_id,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                crop_tile_size=self.crop_tile_size,
            )

            prompt_ids = [bos_id] + pre.tokenized_str
            prompt_mask = [False] + pre.images_seq_mask

            prompt_ids_list.append(prompt_ids)
            prompt_mask_list.append(prompt_mask)
            images_list.append((pre.image_crop, pre.image_ori))
            spatial_crop_list.append(pre.spatial_crop)

        # ---- left-pad the prompt-only sequences (needed for .generate()) ----
        max_prompt = max(len(x) for x in prompt_ids_list)
        prompt_input_ids = torch.full((len(batch_samples), max_prompt), pad_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((len(batch_samples), max_prompt), dtype=torch.long)
        prompt_images_seq_mask = torch.zeros((len(batch_samples), max_prompt), dtype=torch.bool)
        prompt_lengths = []

        for i, ids in enumerate(prompt_ids_list):
            L = len(ids)
            prompt_lengths.append(L)
            prompt_input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            prompt_attention_mask[i, :L] = 1

        for i, mask in enumerate(prompt_mask_list):
            L = len(mask)
            prompt_images_seq_mask[i, :L] = torch.tensor(mask, dtype=torch.bool)

        images_spatial_crop = torch.tensor(spatial_crop_list, dtype=torch.long)

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_images_seq_mask": prompt_images_seq_mask,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "images": images_list,  # list[(crop_tensor, ori_tensor)], one per sample - NOT stacked
            "images_spatial_crop": images_spatial_crop,
            'name': [s['name'] for s in batch_samples],
        }

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=args.trust_remote_code, 
        token=args.hf_token
    )

    dataset = DeepSeekOCRTranslationDataset(
        args.data_dir,
        tokenizer=tokenizer,
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
        'tokenizer':tokenizer,
        'attn_implementation':args.attn_implementation,
        'token':args.hf_token,
        'use_lora':True,
        'lora_r': 8,
        'lora_alpha': 16,
    }
    if args.resume_from_checkpoint:
        model = DeepSeekOCRTranslator.load_from_checkpoint(args.resume_from_checkpoint, map_location='cpu', weights_only=False, **kwargs)
        model = model.to('cuda')
    else:
        model = DeepSeekOCRTranslator(**kwargs)


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
