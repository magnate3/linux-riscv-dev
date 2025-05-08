#!/bin/bash
if [ "$1" != "" ];
then
	destination=""
	if [ "$2" == "" ];
	then
		destination="Connection"
	elif [ "$2" == "E" ];
	then
		destination="EricssonMarker_Connection"
	else
		destination=$2
	fi
	path='/home/love/tofino-master/tofino-master-thesis/Current_project/Evaluation'	
	/usr/local/bin/PcapSplitter -f $path/Flows/Pcap/$1.pcap -o $path/Flows/SplitPcap/newSplit/10.0.0.1/$destination -m connection -i "src host 10.0.0.1"
else
        echo "No input file"
fi
