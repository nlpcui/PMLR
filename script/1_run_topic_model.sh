#!/bin/bash

# topic_wikitext
python -m src.topic_model --dataset=wikitext --num_topics=10 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_wikitext --num_topics=25 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_wikitext --num_topics=50 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_wikitext --num_topics=100 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_wikitext --num_topics=200 --max_iterations=1000 --model_type=lda --task=train

# topic_bill
python -m src.topic_model --dataset=wikitext --num_topics=10 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_bill --num_topics=25 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_bill --num_topics=50 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_bill --num_topics=100 --max_iterations=1000 --model_type=lda --task=train
#python -m src.topic_model --dataset=topic_bill --num_topics=200 --max_iterations=1000 --model_type=lda --task=train