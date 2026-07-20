#/bin/sh

apt update
apt install -y bison flex
apt install -y quagga

# clone latest iproute2
# git clone https://github.com/segment-routing/iproute2.git /tmp/iproute2
# cd /tmp/iproute2

# wget https://mirrors.edge.kernel.org/pub/linux/utils/net/iproute2/iproute2-5.7.0.tar.gz
# tar xfv iproute2-4.18.0.tar.gz
# cd iproute2-4.18.0

# build!
# make && make install
