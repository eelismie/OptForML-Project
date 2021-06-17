#!/bin/sh
# This script goes through different combinations of topologies and node counts in order
# to obtain results that will be plotted in 'notebooks/train_plots.ipynb'

# no. of samples
N=2000
runs=1
nodes=50 #"5 10 50 100"
topos="ring random fc smallworld"
lrs="none" #"none 0.281095"
iters=2000
iid="false"
mix_steps=1 #"1 2 3"

# activate environment
. ./venv/bin/activate

# remove all out CSV files
rm notebooks/out*.csv

for i in $(seq $runs); do
    for n in $nodes; do
        for lr in $lrs; do
            for topo in $topos; do
                for mix_step in $mix_steps; do
                    printf "i : %3d\tnodes : %4d\tlr : %8s\tmix. steps : %3d\ttopo : %8s\n" "$i" "$n" "$lr" "$mix_step" "$topo"
                    csv="notebooks/out.$i.csv"
                    python3 main.py --topo $topo --seed $i --num_samples $N \
                        --batch_size $N --lr $lr --nodes $n --iters $iters \
                        --iid "$iid" --mixing_steps $mix_step --csv "$csv"
                done
            done
        done
    done
done
