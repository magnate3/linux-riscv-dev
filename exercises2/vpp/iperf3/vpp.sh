ip link add name vpp1out type veth peer name vpp1host
ip link set dev vpp1out up
ip link set dev vpp1host up
ip addr add 10.10.1.1/24 dev vpp1host

########### br
brctl addbr br0 
brctl addif br0  vpp1host
brctl addif br0  eth0 
ip l set br0 up
############ start vpp
#./vpp/build-root/build-vpp-native/vpp/bin/vpp  -c  ./vpp/build-root/build-vpp-native/vpp/startup.conf 
########### vpp
#create host-interface name vpp1out
#set int state host-vpp1out up
#show int
#show hardware
#set int ip address host-vpp1out 10.11.11.2/24
#show int addr
############ start vpp
#VCL_CFG=vcl.conf 
#LDP_PATH=./vpp/build-root/build-vpp-native/vpp/lib/libvcl_ldpreload.so
#sh -c "LD_PRELOAD=$LDP_PATH VCL_CONFIG=$VCL_CFG iperf3  -s --bind 10.11.11.2"
