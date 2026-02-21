#!/bin/bash


src=bill
model=lda
prompt_model_id='gpt-4o-mini'
temperature=0.3
num_topics=50
feature_type=topic

# baselines
#for pos_k in 5 10 20
#do
#  for strategy in topk stratified weighted #random
#  do
#    python -m src.label_topic --dataset=${model}_${src} \
#                              --prompt_model_id=${prompt_model_id} \
#                              --temperature=${temperature} \
#                              --samples="output/sampled/label/${src}/${model}${num_topics}_pos${pos_k}_neg0_${strategy}.json" \
#                              --output_file="output/topic_labels/${src}/${model}${num_topics}_pos${pos_k}_neg0_${strategy}.json" \
#                              --saved_features="output/topic_models/${model}_${src}_${num_topics}.json" \
#                              --feature_type="${model}" &
#  done
#done

# ours
for pos_k in 5 10 20
do
  for alpha in 0.01 0.05 0.1
  do
    python -m src.label_topic --dataset=${feature_type}_${src} \
                              --prompt_model_id=${prompt_model_id} \
                              --temperature=${temperature} \
                              --samples="output/sampled/label/${src}/${model}${num_topics}_pos${pos_k}_neg0_dpp_alpha_${alpha}.json" \
                              --output_file="output/topic_labels/${src}/${model}${num_topics}_pos${pos_k}_neg0_dpp_alpha_${alpha}.json" \
                              --saved_features="output/topic_models/${model}_${src}_${num_topics}.json" \
                              --feature_type="${feature_type}" &
  done
done