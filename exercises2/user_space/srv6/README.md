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