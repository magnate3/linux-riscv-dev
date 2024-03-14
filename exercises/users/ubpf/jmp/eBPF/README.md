# eBPF
eBPF DNS filter using XDP
Usage:
	1. clone from github (git clone <repo>)
	2. update libbpf: 
		- git submodule update --init
	3. Make libbpf:
		- cd libbpf/src; make
	4. go to tracker\_v<i> 
		- cd tracker_v<i>
		- make; make load;
		- ./update_xdp_maps
	5. to exit
		- Ctrl+c followed by make unload
	
update_xdp_maps: allows user space control over the maps
in case user wants to forward to specific DNS servers, be adviced to change ip addr
and MAC's in the array inside xdp_network_tracker.c

tracker_v1 : contains a simple XDP filter,
	     packet gets block if length > MAX allowed length
		 	    or if domain isn't allowed

tracker_v2 : in addition to the above filter, redirects the packets to designated NS servers
		

