#!/bin/bash


dataset=wikitext
prompt_model_id='gpt-4o-mini'
temperature=0.5

python -m src.label_topic --dataset=${dataset} \
                          --prompt_model_id=${prompt_model_id} \
                          --temperature=${temperature} \
                          --samples=output/sampled