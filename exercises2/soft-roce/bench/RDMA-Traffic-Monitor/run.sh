set -x
# apt update
# apt install -y build-essential
# apt install -y infiniband-diags
perfquery -V
# pip install confluent-kafka
# pip install ./platformers
# pip install ./platformers/csrc/rotary
#
python -m torch.utils.collect_env
ls /sys/class/infiniband
ibstatus

sudo apt update
sudo apt install -y ethtool
ethtool --version
ethtool -S eth0
# python monitor_ib_traffic.py

# while true; do
#     perfquery -a
#     sleep 1
# done

