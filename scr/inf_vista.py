from configargparse import ArgumentParser
import json
import os
import random
import yaml
from safetensors.torch import load_file
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSeq2SeqLM, 
    NllbTokenizer, 
    AutoModelForImageTextToText,
    AutoProcessor)
from peft import LoraConfig, get_peft_model
import vista_qwen as mt
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
torch.cuda.init()
torch.cuda.set_device(0)
_ = torch.zeros(1).cuda()

class MTDataset(torch.utils.data.Dataset):
    def __init__(self, index_file, processor, tokenizer, target, padding_value):
        self.processor = processor
        self.tokenizer = tokenizer
        self.target = target
        self.padding_value = padding_value

        self.records = []
        source_records_by_dir = {}
        path = mt.resolve_index_files(index_file)[0]
        directory = os.path.dirname(path)
        if directory not in source_records_by_dir:
            source_path = os.path.join(directory, "texts_zho_Hans.json")
            if not os.path.isfile(source_path):
                raise FileNotFoundError(f"Missing source text index: {source_path}")
            source_records_by_dir[directory] = {
                record["image_path"]: record["text"]
                for record in mt.load_texts_json(source_path, "zho_Hans")
            }

        source_records = source_records_by_dir[directory]
        for record in mt.load_texts_json(path, target):
            source_text = source_records.get(record["image_path"])
            if source_text is None:
                raise ValueError(
                    f"No source transcription found for {record['image_path']}"
                )
            self.records.append({**record, "src_txt": source_text})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")

        return {
            "image": image,
            "lang": record["lang"],
            "src_txt": record["src_txt"],
            "name": os.path.basename(record["image_path"]),
        }

    def collate_fn(self, batch):
        images = [sample["image"] for sample in batch]
        vision_inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"]
        grid_thw = vision_inputs["image_grid_thw"].reshape(-1, 3)

        patches_per_image = grid_thw.prod(dim=1)
        if int(patches_per_image.sum()) != pixel_values.shape[0]:
            raise ValueError(
                "Image processor returned pixel values whose patch count does not "
                "match image_grid_thw"
            )
        pixel_attention_mask = (
            torch.arange(patches_per_image.max(), device=pixel_values.device).unsqueeze(0)
            < patches_per_image.to(pixel_values.device).unsqueeze(1)
        )

        src_txt = self.tokenizer(
            [sample["src_txt"] for sample in batch],
            return_tensors="pt",
            padding=True,
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "image_grid_thw": grid_thw,
            "src_txt": src_txt,
            "tgt_langs": [sample["lang"] for sample in batch],
            "name": [sample["name"] for sample in batch],
        }

def parse_args():
    # First parse only --config so the YAML file can be loaded manually.
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to config file.")
    config_args, _ = config_parser.parse_known_args()

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--vision_model", type=str, default='google/siglip2-base-patch16-naflex', help="Path to pretrained vision model.")
    parser.add_argument("--mt_model_path", type=str, default='black-forest-labs/FLUX.2-klein-4B', help="Path to pretrained MT model.")
    parser.add_argument("--checkpoint", type=str, help="Translator checkpoint to load for inference.")
    parser.add_argument("--latents_dir", type=str, help="Directory containing precomputed latents.")
    parser.add_argument("--output_file", type=str, help="File to write the translations.")
    parser.add_argument("--src_lang", type=str, help="Source language token (e.g., <eng_Latn> for English).")
    parser.add_argument("--tgt_lang", type=str, help="Target language token (e.g., <zho_Hans> for Chinese).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for the data loading")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--adapter_args", type=dict, default={"visual_hidden_size":768, "text_hidden_size":1024}, help="Keyword arguments for the image adapter")
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention implementation")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)

    if config_args.config is not None:
        with open(config_args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**config)

    return parser.parse_args()

def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        torch.use_deterministic_algorithms(True)

def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    kwargs = {"attn_implementation":"flash_attention_2", "dtype":torch.float16} if args.flash_attn else {}
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model_path, 
                                                        token=HF_TOKEN, 
                                                        use_safetensors=True,
                                                        **kwargs)
    vision_model = AutoModelForImageTextToText.from_pretrained(args.vision_model, 
                                                      token=HF_TOKEN, 
                                                      use_safetensors=True).model.visual
    processor = AutoProcessor.from_pretrained(args.vision_model, token=HF_TOKEN)
    if args.use_lora:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["qkv", "proj", "linear_fc1", "linear_fc2"]
            )
            vision_model = get_peft_model(vision_model, lora_config)
    # Language tokens
    tokenizer = NllbTokenizer.from_pretrained(args.mt_model_path, 
                                                    token=HF_TOKEN,
                                                    src_lang=args.src_lang,
                                                    tgt_lang=args.tgt_lang,)

    padding_tok = tokenizer.pad_token_id
    kwargs = {
        'mt_model':mt_model,
        'vision_encoder':vision_model,
        'tokenizer':tokenizer,
        'src_lang':args.src_lang,
        'tgt_langs':[args.tgt_lang],
        'encoder_lr':0,
        'adapter_lr':0,
        'decoder_lr':0,
        'min_lr':0,
        'max_steps':0,
        'step_milestones':[0,0,0],
        'mt_loss':0,
        'ctc_loss':0,
        'padding_value':padding_tok,
        'flash_attn':args.flash_attn,
        'adapter_args':args.adapter_args
    }
    if args.checkpoint:
        model = mt.Translator.load_from_checkpoint(args.checkpoint, map_location='cpu', **kwargs)
        model = model.to('cuda')
    else:
        model = mt.Translator(**kwargs)

    trainer = pl.Trainer(
        accelerator='gpu',
        #enable_progress_bar=False,
        deterministic='warn',
        benchmark=False,
    )

    dataset = MTDataset(args.latents_dir, processor = processor, tokenizer = tokenizer, padding_value = padding_tok, target=args.tgt_lang)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    hypotheses = trainer.predict(model, dataloaders=dataloader)

    # records = [{
    #     "idx": batch_idx * len(batch) + sample_idx,
    #     "text": hyp,
    # } for batch_idx, batch in enumerate(hypotheses) for sample_idx, hyp in enumerate(batch)]
    records = [
        {"idx": name,
         "text": hyp} for batch in hypotheses for name, hyp in zip(batch[0], batch[1])
    ]

    with open(args.output_file, 'w') as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    print(f"{len(records)} translations saved to {args.output_file}.")

if __name__ == "__main__":
    main()
