#!/bin/bash

python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-ar/lsvt_test_ar_gemma.json --target_lang arb_Arab --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-be/lsvt_test_be_gemma.json --target_lang ben_Beng --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-en/lsvt_test_en_gemma.json --target_lang eng_Latn --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-es/lsvt_test_es_gemma.json --target_lang spa_Latn --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-fr/lsvt_test_fr_gemma.json --target_lang fra_Latn --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-hi/lsvt_test_hi_gemma.json --target_lang hin_Deva --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-in/lsvt_test_in_gemma.json --target_lang ind_Latn --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-ja/lsvt_test_ja_gemma.json --target_lang jpn_Jpan --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-ru/lsvt_test_ru_gemma.json --target_lang rus_Cyrl --config config/inference_gemma.yaml
python scr/inf_gemma.py --output_file hypotheses/lsvt_zh-ur/lsvt_test_ur_gemma.json --target_lang urd_Arab --config config/inference_gemma.yaml
