#!/bin/bash

python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-ar/lsvt_test_ar_paddleFT.json --target_lang arb_Arab --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-be/lsvt_test_be_paddleFT.json --target_lang ben_Beng --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-en/lsvt_test_en_paddleFT.json --target_lang eng_Latn --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-es/lsvt_test_es_paddleFT.json --target_lang spa_Latn --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-fr/lsvt_test_fr_paddleFT.json --target_lang fra_Latn --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-hi/lsvt_test_hi_paddleFT.json --target_lang hin_Deva --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-in/lsvt_test_in_paddleFT.json --target_lang ind_Latn --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-ja/lsvt_test_ja_paddleFT.json --target_lang jpn_Jpan --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-ru/lsvt_test_ru_paddleFT.json --target_lang rus_Cyrl --config config/inference_paddle.yaml
python scr/inf_paddle.py --output_file hypotheses/lsvt_zh-ur/lsvt_test_ur_paddleFT.json --target_lang urd_Arab --config config/inference_paddle.yaml
