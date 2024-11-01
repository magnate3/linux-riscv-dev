# IPv4 VPN using emulated DT4 through DX4 - single subnet for all remote hosts


Example scenario 3
```
#      arp proxy <------+                                        +------> arp proxy
#                       |                                        |
#                     +------------------+      +------------------+
#                     | |      r0        |      |        r1      | |
#  +-------+          | |                |      |                | |        +-------+
#  |   h0  |          | |                |      |                + |        |  h1   |
#  |       +----------+ veth1      veth2 +------+ veth3      veth4 +--------+       |
#  | veth0 |          |                  |      |                  |        | veth5 |
#  +-------+      10.0.0.254/24    fdff::1/64  fdff::2/64   10.0.0.254/24   +-------+
#                     |                 ^|      |v                 |
# 10.0.0.1/24         +------------------+      +----------+-------+       10.0.0.2/24
#                                       |        |         |
#                                       |        |         |
#                                       |        |         +----------------------------+
#                                       |        |                                      |
#                                       +        +                                      |
#             10.0.0.2 encap segs fc00::2        fc00::2 action End.DX4 nh4 1.1.1.1     |
#                                                                              +        |
#                                                  Routing Table 1111          |        v
#                                                +-------------------+         | rule to 1.1.1.1 lookup 1111
#                                                |                   |         |
#                                                | default dev veth4 | <-------+
#                                                |                   |
#                                                +-------------------+
#
#                                        Routing Table 1111 for End.DT4 emulation
```
