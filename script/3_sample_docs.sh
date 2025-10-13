#!/bin/bash

# wikitext
# top k
#python -m src.sample_docs --dataset=wikitext --strategy=top_k --pos_k=5 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_topk_pos5_neg0.json
#exit 1
#python -m src.sample_docs --dataset=wikitext --strategy=top_k --pos_k=10 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_topk_pos10_neg0.json
#python -m src.sample_docs --dataset=wikitext --strategy=top_k --pos_k=15 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_topk_pos15_neg0.json
#python -m src.sample_docs --dataset=wikitext --strategy=top_k --pos_k=20 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_topk_pos20_neg0.json
python -m src.sample_docs --dataset=wikitext --strategy=top_k --pos_k=50 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_topk_pos50_neg0.json

# weighted
#for i in 1 2 3
#do
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=5 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_weighted_pos5_neg0_${i}.json
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=10 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_weighted_pos10_neg0_${i}.json
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=15 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_weighted_pos15_neg0_${i}.json
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=20 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_weighted_pos20_neg0_${i}.json
#done

python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=50 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_weighted_pos50_neg0_1.json

# random
#for i in 1 2 3
#do
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=5 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_random_pos5_neg0_${i}.json
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=10 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_random_pos10_neg0_${i}.json
##  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=15 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_random_pos15_neg0_${i}.json
#  python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=20 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_random_pos20_neg0_${i}.json
#done

python -m src.sample_docs --dataset=wikitext --strategy=weighted --pos_k=50 --neg_k=0 --topic_model=lda --num_topics=25 --output_path=wikitext_lda_25_random_pos50_neg0_1.json
