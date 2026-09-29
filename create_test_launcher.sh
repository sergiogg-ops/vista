#!/bin/bash

python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ar_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang arb --tgt_code Arab --model facebook/nllb-200-3.3B --batch_size 4
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_be_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang ben --tgt_code Beng --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_es_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang eng --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_es_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang spa --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_fr_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang fra --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_hi_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang hin --tgt_code Deva --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_in_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang ind --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ja_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang jpn --tgt_code Jpan --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ru_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang rus --tgt_code Cyrl --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ur_test_gemma.txt --src_lang zho --src_code Hans --tgt_lang urd --tgt_code Arab --model facebook/nllb-200-3.3B --batch_size 2

python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ar_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Arabic --tgt_code ar --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_be_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Bengali --tgt_code bn --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_es_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Spanish --tgt_code es --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_fr_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang French --tgt_code fr --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_hi_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Hindi --tgt_code hi --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_in_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Indonesian --tgt_code id --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ja_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Japanese --tgt_code ja --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ru_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Russian --tgt_code ru --model translategemma:27b --batch_size 8
python scr/translate.py OCRMT30k/before_tok_bpe/ocr_zh_test.txt OCRMT30k/before_tok_bpe/ocr_ur_test_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Urdu --tgt_code ur --model translategemma:27b --batch_size 8

python scr/translate.py LSVT/test_zh.txt LSVT/test_ar_nllb.txt --src_lang zho --src_code Hans --tgt_lang arb --tgt_code Arab --model facebook/nllb-200-3.3B --batch_size 4
python scr/translate.py LSVT/test_zh.txt LSVT/test_be_nllb.txt --src_lang zho --src_code Hans --tgt_lang ben --tgt_code Beng --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_be_nllb.txt --src_lang zho --src_code Hans --tgt_lang eng --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_es_nllb.txt --src_lang zho --src_code Hans --tgt_lang spa --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_fr_nllb.txt --src_lang zho --src_code Hans --tgt_lang fra --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_hi_nllb.txt --src_lang zho --src_code Hans --tgt_lang hin --tgt_code Deva --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_in_nllb.txt --src_lang zho --src_code Hans --tgt_lang ind --tgt_code Latn --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_ja_nllb.txt --src_lang zho --src_code Hans --tgt_lang jpn --tgt_code Jpan --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_ru_nllb.txt --src_lang zho --src_code Hans --tgt_lang rus --tgt_code Cyrl --model facebook/nllb-200-3.3B --batch_size 2
python scr/translate.py LSVT/test_zh.txt LSVT/test_ur_nllb.txt --src_lang zho --src_code Hans --tgt_lang urd --tgt_code Arab --model facebook/nllb-200-3.3B --batch_size 2

python scr/translate.py LSVT/test_zh.txt LSVT/test_ar_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Arabic --tgt_code ar --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_be_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Bengali --tgt_code bn --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_be_gemma.txt --src_lang Chinese --src_code zh --tgt_lang English --tgt_code en --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_es_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Spanish --tgt_code es --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_fr_gemma.txt --src_lang Chinese --src_code zh --tgt_lang French --tgt_code fr --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_hi_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Hindi --tgt_code hi --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_in_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Indonesian --tgt_code id --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_ja_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Japanese --tgt_code ja --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_ru_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Russian --tgt_code ru --model translategemma:27b --batch_size 8
python scr/translate.py LSVT/test_zh.txt LSVT/test_ur_gemma.txt --src_lang Chinese --src_code zh --tgt_lang Urdu --tgt_code ur --model translategemma:27b --batch_size 8
