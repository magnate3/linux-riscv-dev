#!/bin/bash

cat << EOF > /etc/quagga/zebra.conf
hostname test
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/srv6-1_zebra.conf
hostname srv6-1
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/srv6-2_zebra.conf
hostname srv6-2
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/srv6-3_zebra.conf
hostname srv6-3
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/srv6-4_zebra.conf
hostname srv6-4
password zebra
ip forwarding
EOF


cat << EOF > /etc/quagga/srv6-5_zebra.conf
hostname srv6-5
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/srv6-6_zebra.conf
hostname srv6-6
password zebra
ip forwarding
EOF


cat << EOF > /etc/quagga/ipv6-1_zebra.conf
hostname ipv6-1
password zebra
ip forwarding
EOF

cat << EOF > /etc/quagga/ipv6-2_zebra.conf
hostname ipv6-2
password zebra
ip forwarding
EOF


systemctl restart zebra
systemctl status zebra
