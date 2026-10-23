"""
Fine-tuning script for DeepSeek-OCR-2 (multilingual image -> text translation),
adapted from a Gemma-based Lightning training script.

===============================================================================
READ THIS FIRST - important differences from the Gemma version
===============================================================================
DeepSeek-OCR / DeepSeek-OCR-2 are NOT standard `AutoProcessor` multimodal models.
The Hugging Face repos only ship a custom `AutoTokenizer` + `AutoModel`
(trust_remote_code=True), and all image tiling / tokenization / chat-templating
is done by hand inside the model's own `infer()` method. There is no
`apply_chat_template`, no `AutoImageProcessor`.

This script reimplements that same hand-rolled preprocessing (tiling, image
token counting, the "plain" prompt style) directly, based on the current
deepseek-ai/DeepSeek-OCR `modeling_deepseekocr.py` / `conversation.py`
(deepseek_vl_v2 architecture family, which DeepSeek-OCR-2 shares). Since
DeepSeek-OCR-2 postdates my reliable knowledge, and I could not fetch its
modeling file directly, **please diff this logic against the actual files in
your `deepseek-ai/DeepSeek-OCR-2` checkoint** (image_token id, patch_size,
downsample_ratio, tiling thresholds) before a long training run. A quick way:

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)
    model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)
    print(type(model).__module__)  # find the cached modeling_*.py file on disk and diff it

Known, important consequences of the real architecture (verify these!):
  1. In the public v1 modeling code, the vision encoders (SAM + CLIP) AND the
     projector run inside `torch.no_grad()` even while `self.training=True`.
     That means gradients never reach them - only the language-model decoder,
     token embeddings, and lm_head are actually trainable. `--freeze_vision_encoder`
     is therefore close to a no-op (they're already frozen); it's kept only in
     case DeepSeek-OCR-2 changed this.
  2. The language model is DeepSeek-V2 (multi-head latent attention + MoE MLP),
     not a plain Gemma decoder. Its linear layer names differ
     (q_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj, gate_proj/up_proj/down_proj,
     and MoE "experts.N.*" + a router "gate"). The default LoRA target list below
     is a best-effort guess restricted to attention + dense-MLP projections;
     run with `--list_modules` first to confirm names on your actual checkpoint,
     especially if it uses MoE layers (LoRA-on-experts needs care).
  3. Prompts use the model's "plain" conversation style: no role tags, no
     separators - the literal prompt string (with a `<image>` placeholder) is
     tokenized as-is and the target text is appended directly after it.
===============================================================================
"""

import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Optional
from PIL import Image, ImageOps

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision import transforms
from configargparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sacrebleu import corpus_bleu, corpus_ter
import sys
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
    record = json.loads(raw)
    for sample in record:
        sample['lang'] = lang
        sample['idx'] = os.path.join(os.path.dirname(path), sample['idx'] + '.jpg')
    return record


# --------------------------------------------------------------------------------------
# DeepSeek-OCR image preprocessing (ported from the model's own `infer()` method,
# since there is no AutoProcessor to call). Keep this in sync with the actual
# checkpoint's modeling file - see the module docstring above.
# --------------------------------------------------------------------------------------

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    """Tiles `image` into up to `max_num` square crops of side `image_size`.
    This mirrors the cached DeepSeek-OCR-2 preprocessing: the public infer()
    path uses a 768px crop tile and a 2..6 crop-grid search space.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
_image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])


def text_encode(tokenizer, text: str, bos_id: int, eos_id: int, bos: bool = True, eos: bool = False) -> list[int]:
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]
    return t


def normalize_prompt_template(prompt: str, image_token: str) -> str:
    if prompt.count(image_token) == 0:
        prompt = f"{image_token}\n{prompt}"
        print(f"[info] prompt did not contain '{image_token}'; inserted it at the beginning")
    elif prompt.count(image_token) > 1:
        raise ValueError(f"prompt must contain at most one '{image_token}' placeholder, found {prompt.count(image_token)}")

    if "{lang}" not in prompt:
        print("[info] prompt does not contain '{lang}'; using it as a static instruction")
    return prompt


def validate_prompt_template(prompt: str, image_token: str) -> str:
    return normalize_prompt_template(prompt, image_token)


def resolve_image_token_id(tokenizer, image_token: str, requested_id: int) -> int:
    try:
        detected_id = tokenizer.convert_tokens_to_ids(image_token)
        if detected_id is not None and detected_id != tokenizer.unk_token_id:
            if detected_id != requested_id:
                print(f"[info] tokenizer resolves '{image_token}' -> id {detected_id}, which differs from --image_token_id={requested_id}. Using the tokenizer's id.")
            return detected_id
    except Exception as exc:
        print(f"[warn] could not resolve image token id from tokenizer ({exc}); falling back to --image_token_id={requested_id}")
    return requested_id


@dataclass
class ImagePreprocessResult:
    tokenized_str: list  # token ids for the text+image-placeholder portion (no bos)
    images_seq_mask: list  # bool mask, same length as tokenized_str
    image_ori: torch.Tensor  # (1, 3, base_size, base_size)
    image_crop: torch.Tensor  # (P, 3, crop_tile_size, crop_tile_size) or a (1,3,base,base) zero placeholder
    spatial_crop: list  # [width_crop_num, height_crop_num]


def preprocess_prompt_with_image(
    tokenizer,
    image: Image.Image,
    prompt_text: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    image_token: str,
    image_token_id: int,
    patch_size: int,
    downsample_ratio: int,
    crop_tile_size: int = 640,
) -> ImagePreprocessResult:
    """Builds (token ids, image-token mask, image tensors) for ONE prompt containing
    exactly one `image_token` placeholder, mirroring DeepSeek-OCR's `infer()`.
    """
    assert prompt_text.count(image_token) == 1, "expected exactly one <image> placeholder in the prompt"
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    text_before, text_after = prompt_text.split(image_token)

    tokenized_str = text_encode(tokenizer, text_before, bos_id, eos_id, bos=False, eos=False)
    images_seq_mask = [False] * len(tokenized_str)

    image_crop_tensor = torch.zeros((1, 3, base_size, base_size))
    width_crop_num, height_crop_num = 1, 1

    if crop_mode:
        if image.size[0] <= crop_tile_size and image.size[1] <= crop_tile_size:
            crop_ratio = [1, 1]
            images_crop_raw = []
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=crop_tile_size)

        global_view = ImageOps.pad(image, (base_size, base_size), color=tuple(int(x * 255) for x in IMAGE_MEAN))
        image_ori_tensor = _image_to_tensor(global_view).to(torch.bfloat16).unsqueeze(0)

        width_crop_num, height_crop_num = crop_ratio
        spatial_crop = [width_crop_num, height_crop_num]

        num_queries = math.ceil((crop_tile_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

        tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
        tokenized_image += [image_token_id]

        if width_crop_num > 1 or height_crop_num > 1:
            images_crop_list = [_image_to_tensor(c).to(torch.bfloat16) for c in images_crop_raw]
            image_crop_tensor = torch.stack(images_crop_list, dim=0)
            tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (num_queries * height_crop_num)
    else:
        resized = image.resize((image_size, image_size)) if image_size <= 640 else image
        global_view = ImageOps.pad(resized, (image_size, image_size), color=tuple(int(x * 255) for x in IMAGE_MEAN))
        image_ori_tensor = _image_to_tensor(global_view).to(torch.bfloat16).unsqueeze(0)
        spatial_crop = [1, 1]
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        tokenized_image = ([image_token_id] * num_queries) * num_queries
        tokenized_image += [image_token_id]

    tokenized_str += tokenized_image
    images_seq_mask += [True] * len(tokenized_image)

    tokenized_after = text_encode(tokenizer, text_after, bos_id, eos_id, bos=False, eos=False)
    tokenized_str += tokenized_after
    images_seq_mask += [False] * len(tokenized_after)

    return ImagePreprocessResult(
        tokenized_str=tokenized_str,
        images_seq_mask=images_seq_mask,
        image_ori=image_ori_tensor,
        image_crop=image_crop_tensor,
        spatial_crop=spatial_crop,
    )


class DeepSeekOCRTranslationDataset(Dataset):
    def __init__(
        self,
        index_files: str,
        tokenizer,
        prompt: str = "<image>\n<|grounding|>Translate the text in this image into {lang}. ",
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
        targets = {os.path.basename(file).split('.')[0].removeprefix('texts_'): file for file in index_files}
        if 'zho_Hans' in targets:
            del targets['zho_Hans']
        self.targets = list(targets.keys())
        self.records = sum([load_texts_json(targets[lang], lang) for lang in self.targets],[])
        self.tokenizer = tokenizer
        self.prompt = prompt
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
        image = Image.open(record["id"]).convert("RGB")
        lang = record['lang']

        return {
            "image": image,
            "lang": lang,
            "prompt_text": self.prompt.format(lang=CODE2LANG[lang]),
            "txt_tgt": record[lang],
        }

    def _collate_fn(self, batch_samples):
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id  # DeepSeek tokenizers often don't define a pad token

        txt_tgts = [s["txt_tgt"] for s in batch_samples]

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

            target_ids = text_encode(self.tokenizer, s["txt_tgt"], bos_id, eos_id, bos=False, eos=True)

            full_ids = prompt_ids + target_ids
            full_mask = prompt_mask + [False] * len(target_ids)

            if len(full_ids) > self.max_length:
                # Truncate the TARGET side only; keep the full prompt + image tokens intact.
                keep = self.max_length - len(prompt_ids)
                keep = max(keep, 1)
                full_ids = prompt_ids + target_ids[:keep]
                full_mask = prompt_mask + [False] * len(target_ids[:keep])

            full_ids_list.append(full_ids)
            full_mask_list.append(full_mask)
            prompt_ids_list.append(prompt_ids)
            prompt_mask_list.append(prompt_mask)
            images_list.append((pre.image_crop, pre.image_ori))
            spatial_crop_list.append(pre.spatial_crop)

        # ---- right-pad the full (train) sequences ----
        max_full = max(len(x) for x in full_ids_list)
        input_ids = torch.full((len(batch_samples), max_full), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_samples), max_full), dtype=torch.long)
        images_seq_mask = torch.zeros((len(batch_samples), max_full), dtype=torch.bool)
        labels = torch.full((len(batch_samples), max_full), -100, dtype=torch.long)

        for i, (ids, mask) in enumerate(zip(full_ids_list, full_mask_list)):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            images_seq_mask[i, :L] = torch.tensor(mask, dtype=torch.bool)
            prompt_len = len(prompt_ids_list[i])
            labels[i, prompt_len:L] = torch.tensor(ids[prompt_len:L], dtype=torch.long)

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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images_seq_mask": images_seq_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_images_seq_mask": prompt_images_seq_mask,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "images": images_list,  # list[(crop_tensor, ori_tensor)], one per sample - NOT stacked
            "images_spatial_crop": images_spatial_crop,
            "txt_tgt": txt_tgts,
        }


class DeepSeekOCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        prompt: str,
        val_data_dir: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
        dataset_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.tokenizer = tokenizer
        self.prompt = validate_prompt_template(prompt, dataset_kwargs.get("image_token", "<image>") if dataset_kwargs else "<image>")
        self.dataset_kwargs = dataset_kwargs or {}

    def setup(self, stage: Optional[str] = None):
        self.train_ds = DeepSeekOCRTranslationDataset(
            self.data_dir, self.tokenizer, self.prompt, **self.dataset_kwargs
        )
        self.val_ds = DeepSeekOCRTranslationDataset(
            self.val_data_dir, self.tokenizer, self.prompt, **self.dataset_kwargs
        )

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


class DeepSeekOCRTranslator(pl.LightningModule):
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
        use_lora: bool = False,
        lora_r: float = 8,
        lora_alpha: float = 16,
        lora_target_modules: Optional[list] = None,
        max_new_tokens: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        # NOTE: DeepSeek-OCR / DeepSeek-OCR-2 ship as `AutoModel`, not
        # `AutoModelForMultimodalLM` / `AutoModelForCausalLM` - and they REQUIRE
        # trust_remote_code=True (there is no native transformers implementation).
        # The model card also uses the underscore-prefixed `_attn_implementation`
        # kwarg rather than `attn_implementation`; we mirror that here.
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            #_attn_implementation=attn_implementation,
            use_safetensors=True,
            token=token,
        )
        model_module = sys.modules[type(self.model).__module__]
        model_module.print = lambda *args, **kwargs: None
        self.model = self.model.to(torch.bfloat16)
        self.tokenizer = tokenizer

        inner = getattr(self.model, "model", self.model)
        for attr in ("sam_model", "vision_model", "projector"):
            submodule = getattr(inner, attr, None)
            if submodule is None:
                continue
            trainable_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in submodule.parameters())
            print(f"[info] {attr}: {trainable_params}/{total_params} parameters require grad")

        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e:
                print(f"[warn] gradient_checkpointing_enable failed ({e}); continuing without it.")

        self.hyp, self.tgt = [], []

        # See module docstring: in the released v1 code the vision towers +
        # projector run under torch.no_grad(), so this freeze is likely already
        # the de-facto state. We still expose the flag/hooks in case
        # DeepSeek-OCR-2 changed that.
        inner = getattr(self.model, "model", self.model)
        if freeze_vision_encoder:
            for attr in ("sam_model", "vision_model", "projector"):
                submodule = getattr(inner, attr, None)
                if submodule is not None:
                    for p in submodule.parameters():
                        p.requires_grad = False
                else:
                    print(f"[warn] freeze_vision_encoder: no attribute '{attr}' found on the model; skipping.")

        if use_lora:
            default_targets = [
                "q_proj", "o_proj",
                "kv_a_proj_with_mqa", "kv_b_proj",  # DeepSeek-V2 MLA projections
                "gate_proj", "up_proj", "down_proj",  # dense MLP (non-MoE layers)
            ]
            if not lora_target_modules:
                print("[warn] no explicit --lora_target_modules provided; using a DeepSeek-V2-safe default. Run --list_modules first to confirm the checkpoint names.")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or default_targets,
                lora_dropout=0.05,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, **batch):
        return self.model(**batch)

    def _model_kwargs(self, batch: dict) -> dict:
        input_ids = batch["input_ids"]
        input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds = input_embeds.clone()
        return dict(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            inputs_embeds=input_embeds,
            images=batch["images"],
            images_seq_mask=batch["images_seq_mask"],
            images_spatial_crop=batch["images_spatial_crop"],
        )

    def _step(self, batch: dict, stage: str):
        out = self.model(labels=batch["labels"], **self._model_kwargs(batch))
        self.log(f"{stage}_loss", out.loss, prog_bar=True, sync_dist=True, batch_size=batch["input_ids"].size(0))
        return out.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        generated = self.model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            images=batch["images"],
            images_seq_mask=batch["prompt_images_seq_mask"],
            images_spatial_crop=batch["images_spatial_crop"],
            max_new_tokens=self.hparams.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_lengths = batch["prompt_lengths"].tolist()
        gen_txt = []
        for sample_idx, prompt_len in enumerate(prompt_lengths):
            new_tokens = generated[sample_idx, prompt_len:]
            gen_txt.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        self.hyp.extend(gen_txt)
        self.tgt.extend(batch["txt_tgt"])
        return self._step(batch, "val")
    
    def predict_step(self, batch):
        generated = self.model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            images=batch["images"],
            images_seq_mask=batch["prompt_images_seq_mask"],
            images_spatial_crop=batch["images_spatial_crop"],
            max_new_tokens=self.hparams.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_lengths = batch["prompt_lengths"].tolist()
        gen_txt = []
        for sample_idx, prompt_len in enumerate(prompt_lengths):
            new_tokens = generated[sample_idx, prompt_len:]
            gen_txt.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return batch['name'], gen_txt

    def on_validation_epoch_start(self):
        self.hyp, self.tgt = [], []

    def on_validation_epoch_end(self):
        if not self.hyp:
            return
        bleu = corpus_bleu(self.hyp, [self.tgt])
        self.log('val_BLEU', bleu.score)
        ter = corpus_ter(self.hyp, [self.tgt])
        self.log('val_TER', ter.score)
        n = min(5, len(self.hyp))
        examples = [[self.hyp[idx], self.tgt[idx]] for idx in torch.randint(0, len(self.hyp), (n,),)]
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
    parser = ArgumentParser(description="Fine-tune DeepSeek-OCR-2 for image -> language translation.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", is_config_file=True)

    parser.add_argument("--data_dir", type=str, required=True, help="Output dir of a precompute script (contains texts.json + *.jpg).")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Optional separate precomputed validation dir. If omitted, use a random split of --data_dir instead (not implemented here - pass an explicit dir).")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-OCR-2", help="HF repo id or local path to DeepSeek-OCR-2.")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HF auth token, if the model is private.")

    parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Translate the text in this image into {lang}. ", help="Instruction prompt used for every sample; must contain exactly one '<image>' placeholder and a '{lang}' format slot.")
    parser.add_argument("--image_token", type=str, default="<image>", help="Literal image placeholder token in the prompt string.")
    parser.add_argument("--image_token_id", type=int, default=128815, help="Token id the tokenizer uses for the image placeholder. VERIFY against your DeepSeek-OCR-2 tokenizer - this is the value used by DeepSeek-OCR v1 and may differ.")
    parser.add_argument("--patch_size", type=int, default=16, help="Vision encoder patch size (constant in the released code).")
    parser.add_argument("--downsample_ratio", type=int, default=4, help="Vision-to-language token downsample ratio (constant in the released code).")
    parser.add_argument("--base_size", type=int, default=1024, help="Global-view resolution.")
    parser.add_argument("--image_size", type=int, default=768, help="Used for the non-crop-mode global view resize / token-count formula. NOTE: the crop *tile* size is controlled separately by --crop_tile_size (see docstring).")
    parser.add_argument("--crop_tile_size", type=int, default=768, help="Size of each local crop tile in crop_mode. The cached DeepSeek-OCR-2 preprocessing uses 768px tiles and a 2..6 crop grid.")
    parser.add_argument("--disable_crop_mode", action="store_true", help="Disable dynamic tiling (crop_mode=False); use a single resized global view instead.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total (prompt + target) token length; the target side is truncated to fit.")

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning of the language-model decoder.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_target_modules", nargs="+", default=None, help="Optional explicit LoRA target module names. Use --list_modules first to verify the checkpoint's actual layer names.")
    parser.add_argument("--list_modules", action="store_true", help="Load the model, print all named modules, and exit. Use this to verify LoRA target module names before training.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a single-batch preprocessing smoke test and exit before training.")

    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--monitor_metric", type=str, default="val_loss", help="Metric to monitor for early stopping.")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"], help="Mode for early stopping (min or max).")
    parser.add_argument("--patience", type=int, default=3, help="Number of validation checks to wait before stopping.")

    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to write Lightning checkpoints.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
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
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--freeze_vision_encoder", action="store_true", help="Likely a no-op - see module docstring.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens generated during validation.")
    parser.add_argument("--val_check_interval", type=float, default=0.25)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--wandb_project", type=str, default="deepseekocr-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=args.hf_token)
    args.image_token_id = resolve_image_token_id(tokenizer, args.image_token, args.image_token_id)
    args.prompt = validate_prompt_template(args.prompt, args.image_token)

    if args.list_modules:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, token=args.hf_token)
        for name, _ in model.named_modules():
            print(name)
        return

    dataset_kwargs = dict(
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=not args.disable_crop_mode,
        image_token=args.image_token,
        image_token_id=args.image_token_id,
        patch_size=args.patch_size,
        downsample_ratio=args.downsample_ratio,
        crop_tile_size=args.crop_tile_size,
        max_length=args.max_length,
    )

    if args.smoke_test:
        ds = DeepSeekOCRTranslationDataset(args.data_dir, tokenizer=tokenizer, prompt=args.prompt, **dataset_kwargs)
        if len(ds) == 0:
            raise ValueError("smoke test failed: dataset is empty")
        batch = ds._collate_fn([ds[0]])
        print("[info] smoke test passed")
        print(f"[info] input_ids shape: {tuple(batch['input_ids'].shape)}")
        print(f"[info] attention_mask shape: {tuple(batch['attention_mask'].shape)}")
        print(f"[info] labels shape: {tuple(batch['labels'].shape)}")
        print(f"[info] images count: {len(batch['images'])}")
        return

    datamodule = DeepSeekOCRDataModule(
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        tokenizer=tokenizer,
        prompt=args.prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset_kwargs=dataset_kwargs,
    )

    model = DeepSeekOCRTranslator(
        model_path=args.model_path,
        tokenizer=tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        freeze_vision_encoder=args.freeze_vision_encoder,
        #attn_implementation=args.attn_implementation,
        token=args.hf_token,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        max_new_tokens=args.max_new_tokens,
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
        filename="deepseekocr-{epoch:02d}-{val_loss:.4f}",
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