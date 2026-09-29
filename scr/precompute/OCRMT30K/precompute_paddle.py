import torch
import numpy as np
from safetensors.torch import save_file
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from configargparse import ArgumentParser
from shutil import rmtree
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import os
import cv2
import re
import unicodedata
from dotenv import load_dotenv
import json

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

def load_paddle(
        path: str,
        token: str = None,
        device: str = "cuda"
        ) -> tuple:
    """
    Load only the image processor and model from a specified path.

    :param path: path to the model directory
    :type path: str
    :param token: authentication token for private models
    :type token: str
    :return: (image_processor, model)
    :rtype: tuple[AutoImageProcessor, AutoModel]
    """
    image_processor = AutoImageProcessor.from_pretrained(path, token=token)
    #paddle = AutoModel.from_pretrained(path, token=token).to(device).eval()
    return image_processor#, paddle

def load_nllb(
        path: str,
        device: str = "cuda"
) -> tuple:
    tokenizer = NllbTokenizer.from_pretrained(path, 
                                              use_safetensors=True, 
                                              src_lang="zho_Hans",
                                              tgt_lang="eng_Latn")
    # tokenizer.src_lang = 'zho_Hans'
    # tokenizer.tgt_lang = 'eng_Latn'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        use_safetensors=True
    ).to(device).eval()
    return tokenizer, model

def load_image(path: str) -> Image.Image:
    """
    Load an image from a given path.
    
    :param path: path to the image file
    :type path: str
    :return: loaded image
    :rtype: Image.Image
    """
    return Image.open(path).convert("RGB")

def crop_bounding_box(image: Image.Image, polygon: list, margin: int = 0) -> Image.Image:
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

    # Compute the axis-aligned bounding box from all polygon points
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    x_min, y_min = int(np.floor(x_min)), int(np.floor(y_min))
    x_max, y_max = int(np.ceil(x_max)),  int(np.ceil(y_max))

    # Clamp to image bounds
    img_w, img_h = image.size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return image.crop((x_min, y_min, x_max, y_max))

def compute_samples(
        src_image: Image.Image,
        src_text: list[str],
        bbox: list[list],
        processor: AutoImageProcessor,
        tokenizer: NllbTokenizer,
        tgt_text: dict[str, str] = None,
        device: str = "cuda",
) -> list:
    image = crop_bounding_box(src_image, bbox)
    processed = processor(images=image, return_tensors='pt')
    processed = {k: v.to(device) for k, v in processed.items()}
    sample = {k: v for k, v in processed.items()}

    # Encode all transcriptions in one batch
    tok_src_txts = tokenizer(text=src_text, return_tensors='pt')
    sample['src_txt_tok_ids'] = tok_src_txts.input_ids[0]
    if tgt_text:
        for lang in tgt_text.keys():
            tokenizer.tgt_lang = lang
            tok_tgt_txts = tokenizer(text_target=[tgt_text[lang]], return_tensors='pt')
            sample[f"{lang}_txt_tok_ids"] = tok_tgt_txts.input_ids[0]

    return sample

def parse_args():
    parser = ArgumentParser(description="Precompute latents and text embeddings for Flux2Klein.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", is_config_file=True)
    parser.add_argument("--paddle_path", type=str,help="Path to the SigLip model directory.")
    parser.add_argument("--mt_path", type=str, help="Path to the MT model directory.")
    parser.add_argument("--index_file", help="File that correlates the index and the identifier of the images")
    parser.add_argument("--src_text_file", help="File that contains the transcriptions of the images")
    parser.add_argument("--tgt_text_files", nargs="*", help="Files that contain the translations of the transcriptions of the images")
    parser.add_argument("--target_langs", nargs="+", help="List of target languages corresponding to the translation files.")
    parser.add_argument("--img_path", type=str, help="Path to the directory containing source images.")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata directory containing files with bounding box information.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process, or -1 to process all available samples.")
    parser.add_argument("--add_synthetic", action="store_true", help="Whether to use the translation model to include translate the transcriptions and add them to the dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the precomputed data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def parse_metadata(filename: str) -> tuple[list[list[int]], list[str]]:
    bboxes, texts = [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            # Split on commas, but only consume leading numeric tokens
            # Everything after the last number is the text label
            match = re.match(r'^((?:-?\d+(?:\.\d+)?,)+)(.+)$', line)
            # if not match:
            #     continue
            bboxes.append([float(n) for n in match.group(1).rstrip(",").split(",")])
            texts.append(match.group(2).strip())
    return bboxes, texts

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)   # unify visually identical codepoints
    s = re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', '', s)  # strip all whitespace & zero-width chars
    return s.strip()

def get_bbox(filename, text):
    bboxes, texts = parse_metadata(filename)
    text = normalize(text)
    for bbox, txt in zip(bboxes, texts):
        if normalize(txt) == text:
            return bbox
    return None

def create_name(ini_name):
    path, ext = os.path.splitext(ini_name)
    count = 0
    while os.path.exists(f"{path}_{count}{ext}"):
        count += 1
    return path + "_" + str(count) + ext

def store_data(data, dirname):
    keys = list(data.keys())
    os.mkdir(dirname)
    for k in data.keys():
        if "_txt_tok_ids" in k:
            keys.remove(k)
            filename = os.path.join(dirname, k + '.safetensors')
            save_file({"tgt": data[k]}, filename)
    img_data = {k:data[k] for k in keys}
    filename = os.path.join(dirname, "image.safetensors")
    save_file(img_data, filename)

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    processor = load_paddle(args.paddle_path, TOKEN, device)
    tokenizer, mt_model = load_nllb(args.mt_path)

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
    lang_entries = {lang: [] for lang in args.target_langs + ["zho_Hans"]}

    failures = []
    for idx, name in tqdm(enumerate(identifiers[:num_samples]), desc="Processing batches", total=num_samples):
        identifier = os.path.splitext(name)[0]
        metadata_file = os.path.join(args.metadata_path, f"{identifier}.txt")
        img = load_image(os.path.join(args.img_path,identifier + '.jpg'))
        bbox = get_bbox(metadata_file, src_texts[idx])
        if bbox is None:
            failures.append(idx)
            continue
        source_txt = src_texts[idx].strip()
        target_txt = {lang: texts[idx].strip() for lang, texts in zip(args.target_langs, tgt_texts)} if tgt_texts else {}
        with torch.no_grad():
            sample = compute_samples(
                src_image=img,
                src_text=source_txt,
                tgt_text=target_txt,
                bbox=bbox,
                processor=processor,
                tokenizer=tokenizer,
                device=device
            )
        # Save each sample without batch dimension

        namesave = create_name(os.path.join(args.output_dir, identifier))
        store_data(sample, namesave)

        target_txt["zho_Hans"] = source_txt
        for lang, text in target_txt.items():
            lang_entries[lang].append({"idx": os.path.basename(namesave), "text": text})

    for lang, entries in lang_entries.items():
        with open(os.path.join(args.output_dir, f'texts_{lang}.json'), 'w') as file:
            json.dump(entries, file, ensure_ascii=False, indent=2)
            file.write("\n")
    if failures == 0:
        print("Data processed successfully.")
    else:
        print(f"Processing completed.\nCould not process {len(failures)} samples")
        #print(failures)
    print(f"{num_samples - len(failures)} samples stored in {args.output_dir}")

if __name__ == "__main__":
    main()