#!/bin/bash

dataset=wikitext
num_topics=25
feature_type=topic

python -m output.output --dataset="${dataset}" \
                        --num_topics="${num_topics}" \
                        --saved_features="output/topic_models/lda_${dataset}_${num_topics}.json" \
                        --methods="random,stratified,topk,weighted,dpp_0.05" \
                        --print_recall=1 \
                        --print_all=0