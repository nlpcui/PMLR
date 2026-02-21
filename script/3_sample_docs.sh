#!/bin/bash

model=lda
src=bill
num_features=50
feature_type=topic

# for eval
# stratified
#for strategy in stratified random
#do
#  python -m src.sample --dataset="${model}_${src}" \
#                       --sample_method=${strategy} \
#                       --pos_k=100 \
#                       --neg_k=100 \
#                       --num_features=${num_features} \
#                       --saved_features="output/topic_models/${model}_${src}_${num_features}.json" \
#                       --output_path="output/sampled/eval/${src}/${model}${num_features}_pos100_neg100_${strategy}.json" \
#                       --feature_type=${feature_type}
#done


# labeling
# baseline
#for strategy in stratified random weighted topk
#do
#  for num_sel in 5 10 20
#  do
#    python -m src.sample --dataset="${model}_${src}" \
#                         --sample_method=${strategy} \
#                         --pos_k=${num_sel} \
#                         --neg_k=0 \
#                         --num_features=${num_features} \
#                         --saved_features="output/topic_models/${model}_${src}_${num_features}.json" \
#                         --output_path="output/sampled/label/${src}/${model}${num_features}_pos${num_sel}_neg0_${strategy}.json" \
#                         --feature_type=${feature_type}
#  done
#done

# ours
for num_sel in 5 10 20
do
  for alpha in 0.01 0.05 0.1
  do
    python -m src.dpp_retrieve --dataset=${feature_type}_${src} \
                               --feature_type=${feature_type} \
                               --saved_features="output/topic_models/${model}_${src}_${num_features}.json" \
                               --saved_embeddings="output/embeddings/${model}_${src}_pretrained.json" \
                               --alpha=${alpha} \
                               --k=$num_sel \
                               --output_file="output/sampled/label/${src}/${model}${num_features}_pos${num_sel}_neg0_dpp_alpha_${alpha}.json"
  done
done