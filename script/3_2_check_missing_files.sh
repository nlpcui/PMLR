#!/bin/bash

datasets=("wikitext" "bill")
models=("lda" "ctm" "bertopic")
topic_nums=(25 50 100 200)
pos_k_combos=(3 7 10)
neg_k_combos=(0 3)

missing=0
job=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for k in "${topic_nums[@]}"; do
      for pos_k in "${pos_k_combos[@]}"; do
        for neg_k in "${neg_k_combos[@]}"; do
          for strategy in top_k weighted random; do
            runs=1
            [[ "$strategy" != "top_k" ]] && runs=3
            for run in $(seq 1 $runs); do
              job=$((job+1))
              fname="output/sampled_docs/${dataset}_${model}_${k}_${strategy}_pos${pos_k}_neg${neg_k}_${run}.json"
              if [ ! -f "$fname" ]; then
                echo "Missing: $fname"
                missing=$((missing+1))
              fi
            done
          done
        done
      done
    done
  done
done

echo "Checked $job jobs. Missing $missing files."