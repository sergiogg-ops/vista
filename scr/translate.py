from concurrent.futures import ThreadPoolExecutor
from ollama import chat
from abc import ABC, abstractmethod
from configargparse import ArgumentParser
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("source", help="Source file")
    parser.add_argument("output", help="Output file")
    parser.add_argument("--src_lang", required=True, help="Source language")
    parser.add_argument("--src_code", help="Code of the source language.")
    parser.add_argument("--tgt_lang", required=True, help="Target language")
    parser.add_argument("--tgt_code", help="Code of the target language.")
    parser.add_argument("--model", help="System to use to translate")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for model inference")
    return parser.parse_args()

class Translator(ABC):
    def __init__(self, src_lang, tgt_lang, src_code, tgt_code):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_code = src_code
        self.tgt_code = tgt_code

    @abstractmethod
    def __call__(self, sentence: str) -> str:
        pass

    @abstractmethod
    def batch_translate(self, batch: list[str]) -> list[str]:
        pass

class GemmaTranslator(Translator):
    def __init__(
            self, 
            src_lang, 
            tgt_lang, 
            src_code, 
            tgt_code,
            model = "translategemma"
    ):
        super().__init__(src_lang, tgt_lang, src_code, tgt_code)
        self.model = model
        self.prompt = f"You are a professional {self.src_lang} ({self.src_code}) to {self.tgt_lang} ({self.tgt_code}) translator. Your goal is to accurately convey the meaning and nuances of the original {self.src_lang} text while adhering to {self.tgt_lang} grammar, vocabulary, and cultural sensitivities. Produce only the {self.tgt_lang} translation, without any additional explanations or commentary. Please translate the following {self.src_lang} text into {self.tgt_lang}:\n\n"

    def __call__(self, sentence):
        request = self.prompt + sentence
        response = chat(
            model=self.model,
            messages=[{'role': 'user', 'content': request}],
        )
        return response.message.content
    
    def batch_translate(self, batch):
        batch = [(i, batch[i]) for i in range(len(batch))]

        def translate_one(args):
            return args[0], self(args[1])

        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
            response = list(executor.map(translate_one, batch))
        for i, sent in response:
            batch[i] = sent.strip()
        return batch

class NLLBTranslator(Translator):
    def __init__(
            self, 
            src_lang, 
            tgt_lang, 
            src_code, 
            tgt_code,
            model = "facebook/nllb-200-3.3B",
            num_beams=4
    ):
        super().__init__(src_lang, tgt_lang, src_code, tgt_code)
        nllb_src_code = f"{src_lang}_{src_code}"
        nllb_tgt_code = f"{tgt_lang}_{tgt_code}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = NllbTokenizer.from_pretrained(
            model,
            src_lang = nllb_src_code,
            tgt_lang = nllb_tgt_code
        )
        # self.tokenizer.src_lang = nllb_src_code
        # self.tokenizer.tgt_lang = nllb_tgt_code
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(nllb_tgt_code)
        self.num_beams = num_beams
    
    def __call__(self, sentence):
        inputs = self.tokenizer(text=sentence, return_tensors='pt').to(self.device)
        hyp = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tgt_lang_id,
            num_beams=self.num_beams
        )
        hyp = self.tokenizer.decode(hyp, skip_special_tokens=True)[0]
        return hyp

    def batch_translate(self, batch):
        inputs = self.tokenizer(text=batch, padding=True, return_tensors='pt').to(self.device)
        hyp = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tgt_lang_id,
            num_beams=self.num_beams
        )
        hyp = self.tokenizer.batch_decode(hyp, skip_special_tokens=True)
        return hyp

def load_model(model, **kwargs):
    name = model.lower()
    if "nllb" in name:
        return NLLBTranslator(model=model, **kwargs)
    elif "gemma" in name:
        return GemmaTranslator(model=model, **kwargs)
    else:
        raise ValueError(f"Ilegal model identifier '{name}'.")

def main():
    args = parse_args()

    source_sentences = []
    with open(args.source, 'r') as file:
        for img in json.load(file).values():
            for det in img:
                source_sentences.append({
                    'idx': det['id'],
                    'text': det['transcription']
                })
    
    kwargs = {
        'src_lang':args.src_lang, 
        'tgt_lang':args.tgt_lang, 
        'src_code':args.src_code,
        'tgt_code':args.tgt_code,
        'model':args.model
    }
    translator = load_model(**kwargs)
    hypotheses = []
    for i in tqdm(range(0, len(source_sentences), args.batch_size)):
        batch = source_sentences[i:i+args.batch_size]
        src_txt = [det['text'] for det in batch]
        hyp = translator.batch_translate(src_txt)
        # replace any line-jumps in each hypothesis with a single space
        hypotheses.extend([{
            'idx': batch[j]['idx'],
            'text': hyp[j]
        } for j in range(len(batch))])

    with open(args.output, 'w') as file:
        json.dump(hypotheses, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
