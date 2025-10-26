#!/bin/bash

# # wikitext
# python -m src.topic_model --task=eval --dataset=wikitext --num_topics=25 --model_type=lda 
# python -m src.topic_model --task=eval --dataset=wikitext --num_topics=50 --model_type=lda
# python -m src.topic_model --task=eval --dataset=wikitext --num_topics=100 --model_type=lda
# python -m src.topic_model --task=eval --dataset=wikitext --num_topics=200 --model_type=lda


# # bill
# python -m src.topic_model --task=eval --dataset=bill --num_topics=25 --model_type=lda
# python -m src.topic_model --task=eval --dataset=bill --num_topics=50 --model_type=lda
# python -m src.topic_model --task=eval --dataset=bill --num_topics=100 --model_type=lda
# python -m src.topic_model --task=eval --dataset=bill --num_topics=200 --model_type=lda

# Function to run eval and extract metrics
run_eval() {
    dataset=$1
    num_topics=$2
    model_type=$3
    
    echo "========================================" | tee -a logs/eval/all_metrics.txt
    echo "Evaluating: $model_type $dataset K=$num_topics" | tee -a logs/eval/all_metrics.txt
    echo "========================================" | tee -a logs/eval/all_metrics.txt
    
    python -m src.topic_model --task=eval --dataset=$dataset --num_topics=$num_topics --model_type=$model_type 2>&1 | tee logs/eval/${model_type}_${dataset}_${num_topics}.log
    
    echo "" | tee -a logs/eval/all_metrics.txt
}

# Clear previous results
> logs/eval/all_metrics.txt

# # LDA - wikitext
# run_eval wikitext 10 lda
# run_eval wikitext 25 lda
# run_eval wikitext 50 lda
# run_eval wikitext 100 lda
# run_eval wikitext 200 lda

# # LDA - bill
# run_eval bill 10 lda
# run_eval bill 25 lda
# run_eval bill 50 lda
# run_eval bill 100 lda
# run_eval bill 200 lda

# # CTM - wikitext
# run_eval wikitext 10 ctm
# run_eval wikitext 25 ctm
# run_eval wikitext 50 ctm
# run_eval wikitext 100 ctm
# run_eval wikitext 200 ctm

# # CTM - bill
# run_eval bill 10 ctm
# run_eval bill 25 ctm
# run_eval bill 50 ctm
# run_eval bill 100 ctm
run_eval bill 200 ctm

# BERTOPIC - wikitext
# run_eval wikitext 10 bertopic
# run_eval wikitext 25 bertopic
# run_eval wikitext 50 bertopic
# run_eval wikitext 100 bertopic
# run_eval wikitext 200 bertopic

# # BERTOPIC - bill
# run_eval bill 10 bertopic
# run_eval bill 25 bertopic
# run_eval bill 50 bertopic
# run_eval bill 100 bertopic
# run_eval bill 200 bertopic

echo "All evaluations complete! Check logs/eval/all_metrics.txt"