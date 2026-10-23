import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Optional
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from configargparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sacrebleu import corpus_bleu, corpus_ter
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, if present

CODE2LANG = {
    'eng_Latn': 'English', 
    'arb_Arab': 'Arabic', 
    'ben_Beng': 'Bengali', 
    'spa_Latn': 'Spanish', 
    'fra_Latn': 'French', 
    'hin_Deva': 'Hindi', 
    'ind_Latn': 'Indonesian', 
    'jpn_Jpan': 'Japanese', 
    'rus_Cyrl': 'Russian', 
    'urd_Arab': 'Urdu'
}


def load_texts_json(path: str, lang: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r",\s*\]\s*$", "]", raw.strip())
    record = {sample['idx']: {lang: sample['text']} for sample in json.loads(raw)}
    return record

def merge(*records):
    keys = sum([list(r.keys()) for r in records], [])
    merged = {}
    for idx in keys:
        merged[idx] = {}
        for r in records:
            merged[idx].update(r[idx])
    return merged

class QwenOCRTranslationDataset(Dataset):
    def __init__(
        self, 
        index_files, 
        processor, 
        prompt: str = "Translate the {source_lang} text in this image into {target_lang}:", 
    ):
        records = []
        targets = {os.path.basename(file).split('.')[0].removeprefix('text_'): file for file in index_files}
        if 'zho_Hans' in targets:
            del targets['zho_Hans']
        for lang, file in targets.items():
            lang = os.path.basename(file).split('.')[0].removeprefix('text_')
            data = load_texts_json(file, lang)
            records.append(data)
        self.records = merge(*records)
        self.targets = list(targets.keys())
        self.processor = processor
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        # Pass only raw data; let the collator handle batch-level processing
        record = self.records[idx]
        image = Image.open(record["id"]).convert("RGB")
        lang = random.choice(self.targets) 
        
        return {
            "image": image,
            "lang": lang,
            "txt_tgt": record[lang]
        }
    
    def _collate_fn(self, batch_samples):  
        images = [s["image"] for s in batch_samples]      
        # 1. Build string sequences using the model's chat template
        prompt_texts = []
        full_texts = []
        
        for s in batch_samples:
            user_msg = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.prompt.format(
                            source_lang="Chinese",
                            target_lang=CODE2LANG[s["lang"]]
                        )}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": s["image"]}
                    ],
                }
            ]
            
            # Text layout for validation/generation evaluation
            p_text = self.processor.apply_chat_template(
                user_msg,
                add_generation_prompt=True,
                tokenize=False,
                return_dict=True,
            )
            prompt_texts.append(p_text)
            
            # Text layout for training loops
            full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": s["txt_tgt"]}]}]
            f_text = self.processor.apply_chat_template(
                full_msg, add_generation_prompt=False, tokenize=False
            )
            full_texts.append(f_text)

        # 2. Process Training Batch (Right Padded)
        self.processor.tokenizer.padding_side = "right"
        train_inputs = self.processor(
            text=full_texts, 
            images=images,
            return_tensors="pt", 
            padding="longest"
        )
        
        # Process a right-padded version of prompts to easily calculate where the labels begin
        prompt_inputs_for_loss = self.processor(
            text=prompt_texts, 
            images=images,
            return_tensors="pt", 
            padding="longest"
        )

        # 3. Compute Labeled Targets (Mask out prompt tokens with -100)
        labels = train_inputs["input_ids"].clone()
        for i in range(len(batch_samples)):
            # Count non-padded prompt tokens for this sample to find the target onset split
            prompt_len = prompt_inputs_for_loss["attention_mask"][i].sum().item()
            labels[i, :prompt_len] = -100
            
        # Ensure right-padding tokens are explicitly ignored during backpropagation
        labels[train_inputs["attention_mask"] == 0] = -100

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
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"],
            "labels": labels,
            "prompt_input_ids": val_inputs["input_ids"],
            "prompt_attention_mask": val_inputs["attention_mask"],
            "pixel_values": train_inputs["pixel_values"],
            "txt_tgt": [s["txt_tgt"] for s in batch_samples],
        }

        # Dynamically map configuration arrays (like mm_token_type_ids or image_grid_thw for Gemma 3 VL)
        for k, v in train_inputs.items():
            if k not in ["input_ids", "attention_mask", "pixel_values"]:
                collated_batch[k] = v
                
        for k, v in val_inputs.items():
            if k not in ["input_ids", "attention_mask", "pixel_values"]:
                collated_batch[f"prompt_{k}"] = v

        return collated_batch

class QwenOCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        processor: Any,
        prompt: str,
        val_data_dir: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.processor = processor
        self.prompt = prompt

    def setup(self, stage: Optional[str] = None):

        self.train_ds = QwenOCRTranslationDataset(self.data_dir, self.processor, self.prompt)
        self.val_ds = QwenOCRTranslationDataset(self.val_data_dir, self.processor, self.prompt)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_ds._collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_ds._collate_fn,
            pin_memory=True,
        )

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class QwenOCRVLTranslator(pl.LightningModule):
    def __init__(
        self,
        model_path: str,
        tokenizer, 
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        freeze_vision_encoder: bool = False,
        attn_implementation: str = "sdpa",
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        use_lora: bool = False,
        lora_r: float = 8,
        lora_alpha: float = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            token=token,
        )
        self.tokenizer = tokenizer
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.hyp, self.tgt = [], []

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        if freeze_vision_encoder:
            for p in self.model.model.visual.parameters():
                p.requires_grad = False
        if use_lora:
            self.model.model.language_model = get_peft_model(self.model.model.language_model, lora_config)

    def forward(self, **batch):
        return self.model(**batch)

    def _step(self, batch: dict, stage: str):
        out = self.model(**batch)
        self.log(f"{stage}_loss", out.loss, prog_bar=True, sync_dist=True, batch_size=batch["input_ids"].size(0))
        return out.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        generated = self.model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            mm_token_type_ids=batch.get("prompt_mm_token_type_ids"),
            image_grid_thw= batch.get("prompt_image_grid_thw"),
            pixel_values=batch["pixel_values"],
            max_new_tokens=128,
        )
        
        new_tokens = generated[:, batch["prompt_input_ids"].shape[1]:]
        gen_txt = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        self.hyp.extend(gen_txt)
        self.tgt.extend(batch["txt_tgt"])
        return self._step(batch, "val")

    def predict_step(self, batch):
        generated = self.model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            mm_token_type_ids=batch.get("prompt_mm_token_type_ids"),
            image_grid_thw= batch.get("prompt_image_grid_thw"),
            pixel_values=batch["pixel_values"],
            max_new_tokens=128,
        )
        
        new_tokens = generated[:, batch["prompt_input_ids"].shape[1]:]
        gen_txt = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return batch['name'], gen_txt
    
    def on_validation_epoch_start(self):
        self.hyp, self.tgt = [], []
    
    def on_validation_epoch_end(self):
        bleu = corpus_bleu(self.hyp, [self.tgt])
        self.log('val_BLEU', bleu.score)
        ter = corpus_ter(self.hyp, [self.tgt])
        self.log('val_TER', ter.score)
        examples = [[self.hyp[idx], self.tgt[idx]] for idx in torch.randint(0, len(self.hyp), (5,),)]
        self.logger.log_text(
            key='samples',
            columns=['Hipotesis', 'Ground truth'],
            data=examples
        )
        self.hyp, self.tgt = [], []

    def configure_optimizers(self):
        no_decay = ("bias", "norm", "LayerNorm", "layernorm")
        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay_params if any(nd in name for nd in no_decay) else decay_params).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args():
    parser = ArgumentParser(description="Fine-tune GemmaOCR-VL-1.6 for image -> English translation.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", is_config_file=True)

    parser.add_argument("--data_dir", type=str, nargs='+', required=True, help="Output dir of precompute_Gemma.py (contains texts.json + *.jpg).")
    parser.add_argument("--val_data_dir", type=str, nargs='+', default=None, help="Optional separate precomputed validation dir (same format as --data_dir). If omitted, a random split of --data_dir is used instead.")
    parser.add_argument("--model_path", type=str, default="GemmaGemma/GemmaOCR-VL-1.6", help="HF repo id or local path to GemmaOCR-VL-1.6.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HF auth token, if the model is private.")

    parser.add_argument("--prompt", type=str, default="Translate the text in this image into English:", help="Instruction prompt used for every sample.")
    parser.add_argument("--image_token", type=str, default=None, help="Literal image placeholder token used by the model's chat template. Defaults to processor.image_token if available.")
    parser.add_argument("--merge_size", type=int, default=2, help="Spatial merge size of the vision encoder (used to turn patch grid dims into placeholder-token count). Verify against your model's config.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total (prompt + target) token length; sequences are truncated to this.")

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (r) for low-rank adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for low-rank adaptation.")

    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--monitor_metric", type=str, default="val_loss", help="Metric to monitor for early stopping.")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"], help="Mode for early stopping (min or max).")
    parser.add_argument("--patience", type=int, default=3, help="Number of validation checks to wait before stopping.")

    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to write Lightning checkpoints.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Use the model repo's custom Hub code instead of transformers' native GemmaOCR-VL implementation. Only needed on older transformers versions without native support; leave off on transformers>=5.x, which now ships GemmaOCRVLForConditionalGeneration natively.")
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--val_check_interval", type=float, default=0.25)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--wandb_project", type=str, default="Gemmaocr-vl-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, token=args.hf_token
    )

    datamodule = QwenOCRDataModule(
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        processor=processor,
        prompt=args.prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = QwenOCRVLTranslator(
        model_path=args.model_path,
        tokenizer=processor.tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        freeze_vision_encoder=args.freeze_vision_encoder,
        attn_implementation=args.attn_implementation,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        offline=args.wandb_offline,
        save_dir=args.output_dir,
    )
    wandb_logger.log_hyperparams(vars(args))

    callbacks = [ModelCheckpoint(
        dirpath=args.output_dir,
        filename="Qwenocr-vl-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )]
    if args.early_stopping:
        es_callback = EarlyStopping(monitor=args.monitor_metric, mode=args.monitor_mode, patience=args.patience)
        callbacks.append(es_callback)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()