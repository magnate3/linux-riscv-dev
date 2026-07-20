# XDP-nat64
Implementation of XDP based NAT64.

## steps to run
`sudo make` in nat64 folder

`sudo python3 testbed.py` will setup namespace based topology with NeST

`sudo ip netns exec h1 ping6 64:ff9b::0b00:0102 -s 0` will ping IPv4 domain from IPv6 domain
