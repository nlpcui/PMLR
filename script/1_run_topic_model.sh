#!/bin/bash

# wikitext
#python -m src.topic_model --task=train --dataset=wikitext --num_topics=10 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_wikitext_10_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=25 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_wikitext_25_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=50 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_wikitext_50_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=100 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_wikitext_100_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=200 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_wikitext_200_full.log

# bill
#python -m src.topic_model --task=train --dataset=bill --num_topics=10 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_bill_10_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=25 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_bill_25_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=50 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_bill_50_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=100 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_bill_100_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=200 --max_iterations=1000 --model_type=lda 2>&1 | tee logs/lda_bill_200_full.log


# run ctm for both datasets and different topic numbers
#python -m src.topic_model --task=train --dataset=wikitext --num_topics=10 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_wikitext_10_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=25 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_wikitext_25_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=50 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_wikitext_50_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=100 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_wikitext_100_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=200 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_wikitext_200_full.log

# python -m src.topic_model --task=train --dataset=bill --num_topics=10 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_bill_10_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=25 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_bill_25_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=50 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_bill_50_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=100 --num_epochs=50 --model_type=ctm 2>&1 | tee logs/ctm_bill_100_full.log
python -m src.topic_model --task=train --dataset=bill --num_topics=200  --model_type=ctm 2>&1 | tee logs/ctm_bill_200_full.log


#run bertopic for both datasets
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=10 --model_type=bertopic 2>&1 | tee logs/bertopic_wikitext_10_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=25 --model_type=bertopic 2>&1 | tee logs/bertopic_wikitext_25_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=50 --model_type=bertopic 2>&1 | tee logs/bertopic_wikitext_50_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=100 --model_type=bertopic 2>&1 | tee logs/bertopic_wikitext_100_full.log
# python -m src.topic_model --task=train --dataset=wikitext --num_topics=200 --model_type=bertopic 2>&1 | tee logs/bertopic_wikitext_200_full.log

# python -m src.topic_model --task=train --dataset=bill --num_topics=10 --model_type=bertopic 2>&1 | tee logs/bertopic_bill_10_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=25 --model_type=bertopic 2>&1 | tee logs/bertopic_bill_25_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=50 --model_type=bertopic 2>&1 | tee logs/bertopic_bill_50_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=100 --model_type=bertopic 2>&1 | tee logs/bertopic_bill_100_full.log
# python -m src.topic_model --task=train --dataset=bill --num_topics=200 --model_type=bertopic 2>&1 | tee logs/bertopic_bill_200_full.log