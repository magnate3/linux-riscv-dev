 for veth in $(ifconfig | grep "^s1" | cut -d' ' -f1)
     do 
	ip l del $veth 
     done 
 for veth in $(ifconfig | grep "^s2" | cut -d' ' -f1)
     do 
	ip l del $veth 
     done 
