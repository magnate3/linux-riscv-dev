#!/bin/bash

dir=$1

if [ -z "$dir" ]; then
	echo usage: $0 expt-output-dir
	exit
fi

for d in $dir/*; do
	tx=`cat $d/rate-tx.txt | awk '$3 > 0 { sum += $3; count += 1; } END { print sum/count; }'`;
	rx=`cat $d/rate-rx.txt | awk '$3 > 0 { sum += $3; count += 1; } END { print sum/count; }'`;
	printf "%50s  tx: %10.5f Gbit  rx: %10.5f Gbit\n" $d $tx $rx;
done

