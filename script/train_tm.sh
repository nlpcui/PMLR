#!/bin/bash

# wikitext
python cluster.py --dataset=wikitext --num_topics=25 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=wikitext --num_topics=50 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=wikitext --num_topics=100 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=wikitext --num_topics=200 --max_iterations=1000 --model_type=lda

# bill
#python cluster.py --dataset=bill --num_topics=25 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=bill --num_topics=50 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=bill --num_topics=100 --max_iterations=1000 --model_type=lda
#python cluster.py --dataset=bill --num_topics=200 --max_iterations=1000 --model_type=lda