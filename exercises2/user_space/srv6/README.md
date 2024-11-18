# SRv6 在 5G 移动网络中的部署方式

[SRv6 在 5G 移动网络中的部署方式](https://xie.infoq.cn/article/88a6b2a1e03fbe616ee8c62c3)      
[P4-BMv2-RAN-UPF](https://github.com/guimvmatos/P4-BMv2-RAN-UPF/tree/main)  
[How To run SRv6 Mobile Uplane POC](https://github.com/ebiken/p4srv6/blob/c5049a80ba366f0cacf20b8bfb88b21540150383/archive/demo/srv6/demo1-SRv6MobileUplane-dropin.md)    
[SRv6 uSID (micro SID) implementation on P4](https://github.com/netgroup/p4-srv6-usid/tree/master)   

# Per-interface configuration
Several per-interface sysctls are available to control SRv6 behavior.    

+ net.ipv6.conf.*.seg6_enabled (integer)      

Matching packets for this sysctl are those whose active segment (i.e., IPv6 DA) is local to the Linux node.    
0: Drop ingress SR-enabled packets from this interface.   
1: Accept ingress SR-enabled packets and apply basic SRH processing.   

+ net.ipv6.conf.*.seg6_require_hmac (integer)   

-1: Ignore HMAC field.    
0: Accept SR packets without HMAC, validate SR packets with HMAC.   
1: Drop SR packets without HMAC, validate SR packets with HMAC.   

# SR encapsulation and insertion

```Text

he iproute2 tool is used to add SRH onto packets, as such:

ip -6 route add <prefix> encap seg6 mode <encapmode> segs <segments> [hmac <keyid>] dev <device>

The parameters are defined as follows.

prefix: IPv6 prefix of the route.
encapmode: encap to encapsulate matching packets into an outer IPv6 header containing the SRH, and inline to insert the SRH right after the IPv6 header of the original packet.
segments: comma-separated list of segments. Example: fc00::1,fc42::5.
keyid: HMAC key ID, further explained below.
device: any non-loopback device.
```

#  Source address for SRv6 encapsulations

```Text
When a packet is encapsulated within an outer IPv6 header, a source address must be selected for this outer header. By default, an interface address is selected. To change this default value, use the following command.

ip sr tunsrc set <addr>

If addr is set to ::, then the default behavior is assumed.

```

# HMAC configuration

```
The optional HMAC TLV ensures the authenticity and integrity of its SRH. It contains the HMAC computation of the header, realised using an HMAC key ID. This key ID is mapped to a secret passphrase, used as input to the HMAC function. The mapping of HMAC key IDs are configured with the following command.

ip sr hmac set <keyid> <algorithm>

You will then be prompted to enter the passphrase. Leave blank to remove the mapping. The algorithm field selects the hashing algorithm to use. Available options are sha1 and sha256. For security robustness, we recommend the latter.
```


# Add the routing entry
```
# ip -6 route add fc42::/64 via NH encap seg6 mode encap segs fc00:12,fc00:89
```

```Text
Let's decompose the command:   

fc42::/64: the matching prefix for encapsulation
via NH: the "default" next-hop for the route. It can be the next-hop that would normally be used to forward the traffic for the matching prefix, but it does not really matter as the kernel will restart its routing decision process to route the SR-enabled packet to the first segment. Also, the Linux IPv6 stack requires a route to have a valid IPv6 next-hop in order to support features such as ECMP, in order to avoid issues with routes such as ff00::/8 and fe80::/64 that are automatically assigned to each IPv6-enabled interface.
encap seg6: this tells the kernel to give the packet to the SR-IPv6 subsystem.
mode encap: this specifies the encapsulation mode. Two values are possible: encap creates an outer IPv6 header with the SRH attached, followed by the original, unmodified inner packet. The other value, inline, directly attach the SRH to the original IPv6 packet. The encap mode should be used, unless you know what you are doing.
segs fc00:12,fc00:89: a list of comma-separated segments
Other parameters can be added after the segments list:

hmac KEYID: define an HMAC key ID for the SRH. See the page ConfigureSecurityFeatures for more information.
cleanup: this keyword requires the penultimate segment to strip the SRH from the packet before forwarding it to the last segment. This option should only be used in inline mode.
```
# Route for local packets
If you wish to add a route for locally generated packets, you have to specify the MTU for the route by adding mtu NUMBER at the end of the iproute command. Indeed, there is currently no practical way to automatically set a route MTU on insertion (it is of course possible but not in a proper manner). Basically, you need to take the MTU of the outgoing interface and substract the length of the encapsulation. The formula is the following:   

```
encap_size = (1 - isinline)*40 + 8 + nsegs*16 + ishmac*32
```
Where isinline = 1 if the encap mode is inline, 0 otherwise; and ishmac = 1 if an HMAC is set, 0 otherwise.    

# Configure the virtual tunnel source
When the encap mode is used, an outer IPv6 header is created. The destination address of this header is the first segment, and a source address must be selected. By default, the kernel looks for a usable IPv6 address attached to the outgoing interface. However, this process is expensive, and for performances reasons it may be useful to define a static source address for encapsulated packets. This can be performed with the following command:    
```
# ip sr tunsrc set fc01::1
```
You can display the current source address with:
```
# ip sr tunsrc show
```