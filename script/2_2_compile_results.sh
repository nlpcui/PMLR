echo "=====================================================================" > logs/eval/comparison_table.txt
echo "TOPIC MODEL PERFORMANCE COMPARISON" >> logs/eval/comparison_table.txt
echo "=====================================================================" >> logs/eval/comparison_table.txt
echo "" >> logs/eval/comparison_table.txt

# Function to extract metrics from log file
extract_metrics() {
    model=$1
    dataset=$2
    k=$3
    logfile="logs/eval/${model}_${dataset}_${k}.log"
    
    if [ -f "$logfile" ]; then
        line=$(grep "purity scores" $logfile | tail -1)
        purity=$(echo $line | sed -n "s/.*'purity': \([0-9\.]*\),.*/\1/p")
        inv_purity=$(echo $line | sed -n "s/.*'inverse_purity': \([0-9\.]*\),.*/\1/p")
        harmonic=$(echo $line | sed -n "s/.*'harmonic': \([0-9\.]*\)}.*/\1/p")
        
        # Format to 4 digits
        purity=$(printf "%.4f" "$purity")
        inv_purity=$(printf "%.4f" "$inv_purity")
        harmonic=$(printf "%.4f" "$harmonic")
        
        printf "%-10s %-10s K=%-4s  Purity: %-7s Inv: %-7s Harmonic: %-7s\n" \
               "$model" "$dataset" "$k" "$purity" "$inv_purity" "$harmonic"
    else
        printf "%-10s %-10s K=%-4s  [MISSING]\n" "$model" "$dataset" "$k"
    fi
}

# WikiText Results
echo "───────────────────────────────────────────────────────────────────" >> logs/eval/comparison_table.txt
echo "WIKITEXT DATASET" >> logs/eval/comparison_table.txt
echo "───────────────────────────────────────────────────────────────────" >> logs/eval/comparison_table.txt

for k in 10 25 50 100 200; do
    echo "" >> logs/eval/comparison_table.txt
    echo "K = $k:" >> logs/eval/comparison_table.txt
    extract_metrics "lda" "wikitext" $k >> logs/eval/comparison_table.txt
    extract_metrics "ctm" "wikitext" $k >> logs/eval/comparison_table.txt
    extract_metrics "bertopic" "wikitext" $k >> logs/eval/comparison_table.txt
done

# Bill Results
echo "" >> logs/eval/comparison_table.txt
echo "───────────────────────────────────────────────────────────────────" >> logs/eval/comparison_table.txt
echo "BILL DATASET" >> logs/eval/comparison_table.txt
echo "───────────────────────────────────────────────────────────────────" >> logs/eval/comparison_table.txt

for k in 10 25 50 100 200; do
    echo "" >> logs/eval/comparison_table.txt
    echo "K = $k:" >> logs/eval/comparison_table.txt
    extract_metrics "lda" "bill" $k >> logs/eval/comparison_table.txt
    extract_metrics "ctm" "bill" $k >> logs/eval/comparison_table.txt
    extract_metrics "bertopic" "bill" $k >> logs/eval/comparison_table.txt
done

# Summary statistics
echo "" >> logs/eval/comparison_table.txt
echo "=====================================================================" >> logs/eval/comparison_table.txt
echo "BEST PERFORMANCE PER METRIC" >> logs/eval/comparison_table.txt
echo "=====================================================================" >> logs/eval/comparison_table.txt

cat logs/eval/comparison_table.txt

echo ""
echo "✅ Results saved to logs/eval/comparison_table.txt"