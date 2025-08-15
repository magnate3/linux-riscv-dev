#!/bin/bash
for (( i=1; i <= 16; i++ ))    
do
    file=test_result_c${i}_bw 
    echo $file 
    rm ../multiple-ib-process/logs/$file
done
