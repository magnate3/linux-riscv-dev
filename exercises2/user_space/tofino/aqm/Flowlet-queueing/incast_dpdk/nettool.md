source enviroment.sh
lspci -vvv |grep Ethernet
ifconfig
ethtool -i eth1
sudo ifconfig eth1 down
#修改tools.sh的PCIPATH
#tools -> usertools
#drive ie40
./tools.sh setup_dpdk

#解绑
#i40e->igb
./tools.sh unbind_dpdk
ifconfig eth1 up


