#! /bin/sh

ARRAY=(1000 2000 50000 100000 200000 300000 415000)

for use_epoch in ${ARRAY[@]}; do
    nohup python -u /Home/ichikawa/random_feedback/LCP/linear_classifier_probe.py --use_epoch $use_epoch --iter_per_epoch 10000 > "../1009/lafs_LCP_"$use_epoch".log"

done