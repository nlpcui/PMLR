#!/bin/bash

# wikitext
python -m src.topic_model --task=eval --dataset=wikitext --num_topics=25 --model_type=lda
python -m src.topic_model --task=eval --dataset=wikitext --num_topics=50 --model_type=lda
python -m src.topic_model --task=eval --dataset=wikitext --num_topics=100 --model_type=lda
python -m src.topic_model --task=eval --dataset=wikitext --num_topics=200 --model_type=lda


# bill
python -m src.topic_model --task=eval --dataset=bill --num_topics=25 --model_type=lda
python -m src.topic_model --task=eval --dataset=bill --num_topics=50 --model_type=lda
python -m src.topic_model --task=eval --dataset=bill --num_topics=100 --model_type=lda
python -m src.topic_model --task=eval --dataset=bill --num_topics=200 --model_type=lda