#!/bin/bash

topic_model=lda
num_topics=25
sel_func="random"
samples_per_topic=20
task=consistency
dataset="wikitext"

task=consistency
prompt_model_id='gpt-4o-mini'
temperature=0.3
eval_sample_per_cluster=50

python -m evaluate --dataset=${dataset} \
                   --saved_weights="output/topic_models/${topic_model}_${dataset}_${num_topics}.json" \
                   --saved_annotations="output/topic_labels/${topic_model}_${num_topics}_${sel_func}_${samples_per_topic}.json" \
                   --prompt_model_id=${prompt_model_id} \
                   --temperature=${temperature} \
                   --eval_sample_per_cluster=${eval_sample_per_cluster} \
                   --output_path="output/eval_result/${dataset}/${task}/${topic_model}_T${num_topics}_K${samples_per_topic}_E${eval_sample_per_cluster}.json" \
                   --task=$task
