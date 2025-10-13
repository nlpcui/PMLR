#!/bin/bash

#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos5_neg0.json'
#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos10_neg0.json'
##python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos15_neg0.json'
#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos20_neg0.json'
python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos50_neg0.json'

#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_weighted_pos5_neg0_1.json'
#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_weighted_pos10_neg0_1.json'
##python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos15_neg0.json'
#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_weighted_pos20_neg0_1.json'
python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_weighted_pos50_neg0_1.json'

python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_random_pos5_neg0_1.json'
python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_random_pos10_neg0_1.json'
#python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_topk_pos15_neg0.json'
python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_random_pos20_neg0_1.json'
python -m src.eval_label --dataset=wikitext --saved_weights='output/topic_models/lda_wikitext_25.json' --topic_annotations='output/topic_labels/wikitext/wikitext_lda_25_random_pos50_neg0_1.json'


