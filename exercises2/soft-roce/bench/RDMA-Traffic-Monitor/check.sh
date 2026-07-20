# check roce driver version
# ofed_info -s
nvidia-smi
# show_gids

# device info
lspci | grep -i ethernet

# ethtool
apt update
apt install -y ethtool
ethtool --version

python monitor_roce_traffic.py
