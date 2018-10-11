#! /bin/sh

ARRAY=(50 80 100 200 300 400 500)

for batch_size in ${ARRAY[@]}; do
  echo $batch_size
  nohup python -u /Home/ichikawa/random_feedback/fashion_mnist/fashion_dfa.py --batch_size $batch_size > "../1010/batch_size_"$batch_size"fashion_mnist.log"

done
