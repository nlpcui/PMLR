#!/bin/bash

src=wikitext
model=sae
prompt_model_id='gpt-4o-mini'
temperature=0.3
num_topics=256
feature_type=sae

# baseline
#for num_label_sample in 5 10 20
#do
#  for strategy in topk stratified random weighted
#  do
#  python -m src.eval_label --dataset="${feature_type}_${src}" \
#                           --feature_type=${feature_type} \
#                           --prompt_model_id=${prompt_model_id} \
#                           --temperature=${temperature} \
#                           --saved_features="output/topic_models/${model}_${src}_${num_topics}.json" \
#                           --topic_annotations="output/topic_labels/${src}/${model}${num_topics}_pos${num_label_sample}_neg0_${strategy}.json" \
#                           --eval_samples="output/sampled/eval/${src}/${model}${num_topics}_pos100_neg100_stratified.json" \
#                           --output_file="output/eval_result/${src}/${model}${num_topics}_label_${strategy}_pos${num_label_sample}_neg0_eval_stratified_pos100_neg100.json" &
#  done
#done


# ours
for num_label_sample in 5 10 20
do
  for alpha in 0.01 0.05 0.1
  do
    python -m src.eval_label --dataset="${feature_type}_${src}" \
                             --feature_type=${feature_type} \
                             --prompt_model_id=${prompt_model_id} \
                             --temperature=${temperature} \
                             --saved_features="output/topic_models/${model}_${src}_${num_topics}.json" \
                             --topic_annotations="output/topic_labels/${src}/${model}${num_topics}_pos${num_label_sample}_neg0_dpp_alpha_${alpha}.json" \
                             --eval_samples="output/sampled/eval/${src}/${model}${num_topics}_pos100_neg100_stratified.json" \
                             --output_file="output/eval_result/${src}/${model}${num_topics}_label_dpp_${alpha}_pos${num_label_sample}_neg0_eval_stratified_pos100_neg100.json" &
  done
done


