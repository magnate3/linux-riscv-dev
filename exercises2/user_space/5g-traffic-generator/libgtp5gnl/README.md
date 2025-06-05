
# ko

```
root@ubuntux86:# modprobe udp_tunnel
root@ubuntux86:# insmod  gtp5g.ko 
root@ubuntux86:# 
```

```
root@ubuntux86:# ./gtp5g-tunnel list far
root@ubuntux86:# ./gtp5g-tunnel list ger
root@ubuntux86:# 

```


```
sudo gtp5g-tunnel add far   gtp5gtest       1           --action 2
#                            gtplink      far-id          action参数，取值貌似1 2 3，目前就2能用

sudo gtp5g-tunnel add far gtp5gtest 2 --action 2   --hdr-creation    0     89   ${IUPF_IP_OUT}      2152
#					            gtp头信息        描述   TEID + 下一跳ip          gtp的端口
sudo gtp5g-tunnel add pdr gtp5gtest   1     --pcd 1  --hdr-rm 0   --ue-ipv4 ${UE_IP}    --f-teid 88 ${AUPF_IP}  --far-id 1
#				    pdr-id   优先级    去除包头       ue-ip                     TEID + 本地gtp接收ip     采用哪个far
sudo gtp5g-tunnel add pdr gtp5gtest 2 --pcd 2 --ue-ipv4 ${UE_IP} --far-id 2 --gtpu-src-ip=${AUPF_IP}
#									    记录从自己哪个ip发出去的
func usage(prog string) {
	fmt.Fprintf(os.Stderr, `Usage:
    %v <add|mod> <pdr|far|qer> <ifname> <oid> [<options>...]
    %v <delete|get> <pdr|far|qer> <ifname> <oid>
    %v list <pdr|far|qer>

OID format:
    <id>
    or
    <seid>:<id>

PDR Options:
    --pcd <precedence>
    --hdr-rm <outer-header-removal>
    --far-id <existed-far-id>
    --ue-ipv4 <pdi-ue-ipv4>
    --f-teid <i-teid> <local-gtpu-ipv4>
    --sdf-desp <description-string>
    --sdf-tos-traff-cls <tos-traffic-class>
    --sdf-scy-param-idx <security-param-idx>
    --sdf-flow-label <flow-label>
    --sdf-id <id>
    --qer-id <id>
    --gtpu-src-ip <gtpu-src-ip>
    --buffer-usock-path <AF_UNIX-sock-path>

FAR Options:
    --action <apply-action>
    --hdr-creation <description> <o-teid> <peer-ipv4> <peer-port>
    --fwd-policy <mark set in iptable>

QER Options:
    --gate-status <gate-status>
    --mbr-ul <mbr-uplink>
    --mbr-dl <mbr-downlink>
    --gbr-ul <gbr-uplink>
    --gbr-dl <gbr-downlink>
    --qer-corr-id <qer-corr-id>
    --rqi <rqi>
    --qfi <qfi>
    --ppi <ppi>
`, prog, prog, prog)
}
```