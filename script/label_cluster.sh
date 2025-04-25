#!/bin/bash

dataset=wikitext
samples_per_topic=20
topic_model=lda
num_topics=25
prompt_model_id='gpt-4o-mini'
sel_func=top_k
temperature=0.5

python labeling.py --dataset=${dataset} \
                   --sel_func=${sel_func} \
                   --k=${samples_per_topic} \
                   --prompt_model_id=${prompt_model_id} \
                   --temperature=${temperature} \
                   --saved_weights="output/topic_models/${topic_model}_${dataset}_${num_topics}.json" \
                   --output_path="output/topic_labels/${topic_model}_${num_topics}_${sel_func}_${samples_per_topic}.json"
