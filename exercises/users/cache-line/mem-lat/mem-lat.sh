#!/bin/sh
#set -x
work=./mem-lat
buffer_size=1
node=$1
mem=$2
for i in `seq 1 15`; do
   #echo $i
   #echo $buffer_size
   taskset -ac 1 $work -b $buffer_size -s 64
   buffer_size=$(($buffer_size*2))
done
