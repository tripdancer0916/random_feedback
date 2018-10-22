#! /bin/sh

ARRAY=(1 2 5 10 20 50 100 200 300 400 500 600 700 800 900 999)

for use_epoch in ${ARRAY[@]}; do
    nohup python -u /Home/ichikawa/random_feedback/LCP/linear_classifier_probe.py --use_epoch $use_epoch --iter_per_epoch 10000 > "./1022/lafs_LCP_"$use_epoch".log"

done
