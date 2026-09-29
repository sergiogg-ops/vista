#!/bin/bash

python scr/inf_vista.py --output_file hypotheses/lsvt_zh-ar/lsvt_test_ar_vista.json --tgt_lang arb_Arab --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-be/lsvt_test_be_vista.json --tgt_lang ben_Beng --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-en/lsvt_test_en_vista.json --tgt_lang eng_Latn --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-es/lsvt_test_es_vista.json --tgt_lang spa_Latn --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-fr/lsvt_test_fr_vista.json --tgt_lang fra_Latn --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-hi/lsvt_test_hi_vista.json --tgt_lang hin_Deva --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-in/lsvt_test_in_vista.json --tgt_lang ind_Latn --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-ja/lsvt_test_ja_vista.json --tgt_lang jpn_Jpan --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-ru/lsvt_test_ru_vista.json --tgt_lang rus_Cyrl --config config/inference_vista.yaml
python scr/inf_vista.py --output_file hypotheses/lsvt_zh-ur/lsvt_test_ur_vista.json --tgt_lang urd_Arab --config config/inference_vista.yaml
