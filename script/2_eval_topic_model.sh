#!/bin/bash

# topic_wikitext

# 0.09599201463822433
python -m src.topic_model --task=eval --dataset=wikitext --num_topics=10 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_wikitext --num_topics=25 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_wikitext --num_topics=50 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_wikitext --num_topics=100 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_wikitext --num_topics=200 --model_type=lda


# topic_bill
#python -m src.topic_model --task=eval --dataset=topic_bill --num_topics=25 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_bill --num_topics=50 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_bill --num_topics=100 --model_type=lda
#python -m src.topic_model --task=eval --dataset=topic_bill --num_topics=200 --model_type=lda