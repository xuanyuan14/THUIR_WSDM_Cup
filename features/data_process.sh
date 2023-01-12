#!/bin/bash
for i in $(seq 0 9)
do
  nohup python -u data_process.py $i > log/data_process_$i.log 2>&1 &
done