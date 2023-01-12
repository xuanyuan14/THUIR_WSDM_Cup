#!/bin/bash
for k1 in $(seq 0.5 0.1 1.0)
do
    for b in $(seq 0.3 0.1 0.9)
    do
      python bm25_metric_cj.py --k1 ${k1} --b ${b}
    done
done