#!/bin/bash

export portno=55205
export bportno=77808
for i in {1..9}
do
	ssh ms130$i "source killprocess"
done

echo "kill process"
sleep 5
for i in {2..9}
do
	ssh ms130$i "./timeline_rone  -p $portno -b $bportno &" &
done

echo "server started"


#./timeline_rone -h 10.10.67.2,10.10.67.3,10.10.67.4,10.10.67.5,10.10.67.6,10.10.67.7,10.10.67.8,10.10.67.9 -c 1 -M 8 -n 8192 -m 262144   -p $portno -b $bportno > log
