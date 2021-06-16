#!/bin/sh
# This script goes through different combinations of topologies and node counts in order
# to obtain results that will be plotted in 'notebooks/train_plots.ipynb'

# no. of samples
N=2000
runs=5
nodes="1 10 100"
topos="fc ring"
lr=0.281095
iters=100
iid="false"

# activate environment
. ./venv/bin/activate

# remove all out CSV files
rm notebooks/out*.csv

for i in $(seq $runs); do
    for n in $nodes; do
        for topo in $topos; do
            printf "i : %3d\tnodes : %4d\ttopo : %8s\n" "$i" "$n" "$topo"
            csv="notebooks/out.$i.csv"
            python3 main.py --topo $topo --seed $i --num_samples $N \
                --batch_size $N --lr $lr --nodes $n --iters $iters \
                --iid "$iid" --csv "$csv"
        done
    done
done
