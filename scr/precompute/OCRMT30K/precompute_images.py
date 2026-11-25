import torch
import numpy as np
from safetensors.torch import save_file
from transformers import AutoModel, AutoImageProcessor
from PIL import Image, ImageDraw
from configargparse import ArgumentParser
from shutil import rmtree
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import os
import cv2
import re
import unicodedata
import json

def load_image(path: str) -> Image.Image:
    """
    Load an image from a given path.
    
    :param path: path to the image file
    :type path: str
    :return: loaded image
    :rtype: Image.Image
    """
    return Image.open(path).convert("RGB")

def _polygon_points(polygon: list, margin: int = 0) -> list[tuple[float, float]]:
    points = np.array(polygon, dtype=np.float32)

    if points.ndim == 1:
        # Flat list like [x0, y0, x1, y1, ...] — reshape to (N, 2)
        points = points.reshape(-1, 2)

    if margin > 0:
        center = points.mean(axis=0)
        directions = points - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # Avoid division by zero for points exactly at the center
        norms = np.where(norms == 0, 1, norms)
        points = points + (directions / norms) * margin

    return [tuple(p) for p in points.tolist()]

def mask_other_bounding_boxes(
        image: Image.Image,
        target_polygon: list,
        other_polygons: list[list],
        fill: tuple[int, int, int] = (0, 0, 0),
        margin: int = 0,
) -> Image.Image:
    """
    Paint every polygon in ``other_polygons`` with ``fill`` while keeping the
    region of ``target_polygon`` untouched (even where boxes overlap).
    """
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for polygon in other_polygons:
        draw.polygon(_polygon_points(polygon, margin), fill=fill)

    # Restore the target region in case another box overlapped it
    target_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(target_mask).polygon(_polygon_points(target_polygon), fill=255)
    masked.paste(image, (0, 0), target_mask)

    return masked

def compute_samples(
        src_image: Image.Image,
        src_text: list[str],
        tgt_text: list[str],
        bbox: list,
        other_bboxes: list[list],
        processor: AutoImageProcessor,
        tokenizer: NllbTokenizer,
        mt_model: AutoModel = None,
        device: str = "cuda",
) -> list:
    image = mask_other_bounding_boxes(src_image, bbox, other_bboxes)
    processed = processor(images=image, return_tensors='pt')
    processed = {k: v.to(device) for k, v in processed.items()}

    # Encode all transcriptions in one batch
    tok_src_txts = tokenizer(text=src_text, return_tensors='pt')

    # Encode the cropped image with SigLip's vision encoder only
    # img_lats = siglip.vision_model(
    #     pixel_values=processed['pixel_values'],
    #     attention_mask=processed.get('pixel_attention_mask'),
    #     spatial_shapes=processed.get('spatial_shapes')
    # ).last_hidden_state.cpu()

    # Encode all translations in one batch
    sample = {k: v[0] for k, v in processed.items()}
    sample['src_txt_tok_ids'] = tok_src_txts.input_ids[0]
    if tgt_text:
        tok_tgt_txts = tokenizer(text_target=tgt_text, return_tensors='pt')
        sample['tgt_txt_tok_ids'] = tok_tgt_txts.input_ids[0]

        if not mt_model is None:
            syn_tok_tgt_txts = mt_model.generate(
                input_ids=tok_src_txts.input_ids.to(device),
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            )
            sample['syn_tgt_txt_tok_ids'] = syn_tok_tgt_txts[0][1:]

    return sample

def parse_args():
    parser = ArgumentParser(description="Precompute latents and text embeddings for Flux2Klein.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", is_config_file=True)
    parser.add_argument("--mt_path", type=str, help="Path to the MT model directory.")
    parser.add_argument("--index_file", help="File that correlates the index and the identifier of the images")
    parser.add_argument("--src_text_file", help="File that contains the transcriptions of the images")
    parser.add_argument("--tgt_text_files", nargs="+", help="File that contains the translations of the transcriptions of the images")
    parser.add_argument("--target_langs", nargs="+", help="List of target languages corresponding to the translation files.")
    parser.add_argument("--img_path", type=str, help="Path to the directory containing source images.")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata directory containing files with bounding box information.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process, or -1 to process all available samples.")
    parser.add_argument("--add_synthetic", action="store_true", help="Whether to use the translation model to include translate the transcriptions and add them to the dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the precomputed data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')

def parse_metadata(filename: str) -> tuple[list[list[int]], list[str]]:
    bboxes, texts = [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",")
            # Count the leading numeric tokens (the last part is always text)
            n_nums = 0
            while n_nums < len(parts) - 1 and NUMBER_RE.fullmatch(parts[n_nums]):
                n_nums += 1
            # Labels may start with numbers too (e.g. "203,900元"), so only
            # keep a multiple of 4 values (pairs of points, quads or larger
            # polygons) as coordinates; the leftover numbers belong to the text
            n_coords = n_nums - n_nums % 4
            bboxes.append([float(n) for n in parts[:n_coords]])
            texts.append(",".join(parts[n_coords:]).strip())
    return bboxes, texts

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)   # unify visually identical codepoints
    s = re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', '', s)  # strip all whitespace & zero-width chars
    return s.strip()

def get_bbox(filename, text):
    """
    Return ``(target_bbox, other_bboxes)`` where ``target_bbox`` is the first
    box whose label matches ``text`` and ``other_bboxes`` are all the rest.
    Returns ``(None, None)`` if no box matches.
    """
    bboxes, texts = parse_metadata(filename)
    text = normalize(text)
    for i, txt in enumerate(texts):
        if normalize(txt) == text:
            return bboxes[i], bboxes[:i] + bboxes[i + 1:]
    return None, None

SRC_LANG = "zho_Hans"

def write_json(path: str, entries: list[dict]) -> None:
    """
    Write ``entries`` (dicts with ``idx`` and ``text``) as a JSON list.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def format_bbox(bbox: list) -> str:
    """Comma-separated layout of the raw metadata ("x0,y0,x1,y1,...")."""
    return [int(v) if float(v).is_integer() else str(v) for v in bbox]

def create_name(ini_name):
    path, ext = os.path.splitext(ini_name)
    count = 0
    while os.path.exists(f"{path}_{count}{ext}"):
        count += 1
    return path + "_" + str(count) + ext

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #input("Output directory already exists. Press Enter to empty it or Ctrl+C to abort.")
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)

    with open(args.index_file, 'r') as file:
        identifiers = file.readlines()
    with open(args.src_text_file, 'r') as file:
        src_texts = file.readlines()
    if args.tgt_text_files:
        tgt_texts = []
        for tgt_file in args.tgt_text_files:
            with open(tgt_file, 'r') as file:
                tgt_texts.append(file.readlines())
    else:
        tgt_texts = None

    num_samples = min(args.num_samples, len(identifiers)) if args.num_samples > 0 else len(identifiers)
    lang_entries = {lang: [] for lang in args.target_langs + [SRC_LANG]}
    metadata_entries = []

    failures = 0
    for idx, name in tqdm(enumerate(identifiers[:num_samples]), desc="Processing batches", total=num_samples):
        identifier = os.path.splitext(name)[0]
        metadata_file = os.path.join(args.metadata_path, f"{identifier}.txt")
        os.system(f"cp {os.path.join(args.img_path,identifier + '.jpg')} {os.path.join(args.output_dir, identifier + '.jpg')}")
        #img = load_image(os.path.join(args.img_path,identifier + '.jpg'))
        bbox, other_bboxes = get_bbox(metadata_file, src_texts[idx])
        print(type(bbox), type(other_bboxes))
        if bbox is None:
            failures += 1
            continue
        source_txt = src_texts[idx].strip()
        target_txt = {lang: texts[idx].strip() for lang, texts in zip(args.target_langs, tgt_texts)} if tgt_texts else None
        #image = mask_other_bounding_boxes(img, bbox, other_bboxes)
        # Save each sample without batch dimension
        
        #namesave = create_name(os.path.join(args.output_dir, f"{identifier}.jpg"))
        #image.save(namesave)

        for lang, text in target_txt.items():
            lang_entries[lang].append({"idx": identifier, "text": text})
        metadata_entries += [{
            "idx": identifier,
            # Path of the (masked) image
            "image": os.path.join(args.output_dir, f"{identifier}.jpg"),
            # Bounding box of the source (Chinese) text, same comma-separated
            # layout as the raw metadata ("x0,y0,x1,y1,...")
            "bbox": bbox,
            # Bounding boxes of the other texts, masked out in the image
            "others": other_bboxes,
            "source_text": source_txt,
            "target_text": text,
            "source_lang": SRC_LANG,
            "target_lang": lang
        } for lang, text in target_txt.items()]

    for lang, entries in lang_entries.items():
        write_json(os.path.join(args.output_dir, f'texts_{lang}.json'), entries)
    write_json(os.path.join(args.output_dir, 'metadata.json'), metadata_entries)
    if failures == 0:
        print("Data processed successfully.")
    else:
        print(f"Processing completed.\nCould not process {failures} samples")
    print(f"{num_samples} samples stored in {args.output_dir}")

if __name__ == "__main__":
    main()
