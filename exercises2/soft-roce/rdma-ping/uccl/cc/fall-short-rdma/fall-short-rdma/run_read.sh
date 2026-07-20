#!/bin/bash

requests=5000000


	for signal in "1" #"0"
	do 
#	    for buff in "4" "8" "16" "32" "64" "128" "256" "512" "1024" "2048" "4096"
#	    do
		foldname=$signal
		rm -r $foldname
		mkdir $foldname
		killprocess main
		sleep 5	
		for cpuID in {0..7}
		do
		    ./main -i $cpuID -p 5528$cpuID  &
		    pids="$pids $!"
		done
	
		for i in {2..9}
		do
		    portno=$(( $i-2 ))  
		    ssh 10.10.1.$i "cd rdmaServer/read; ./main -p 5528$portno -h 10.10.1.1 -c 1 -n $requests &" & 
		done
		sleep 80
	
		for i in {2..9}
		do
			node=$(( $i-1 ))
		    scp 10.10.1.$i:~/rdmaServer/read/request_latency $foldname/node$node\_request_latency
		    scp 10.10.1.$i:~/rdmaServer/read/tput $foldname/node$node\_tput
		    ssh 10.10.1.$i "rm ~/rdmaServer/read/request_latency; rm ~/rdmaServer/read/tput"
		done

#	    done
	    
	done
