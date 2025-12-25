#!/bin/bash

VIDEO_FOLDER="$1"
NUM_CLIENTS="$2"

VIDEOS=($VIDEO_FOLDER/*)
NUM_VIDEOS=${#VIDEOS[@]}

for i in $(seq 1 $NUM_CLIENTS);
do
    idx=$(($i % ${NUM_VIDEOS}))
    video=${VIDEOS[$idx]}
    echo $i $video
    python client.py --video "$video" &
done