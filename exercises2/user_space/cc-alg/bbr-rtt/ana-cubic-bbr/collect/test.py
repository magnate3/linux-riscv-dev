import re

#text = "ESTAB 0      2364360 10.22.116.220:33024 10.22.116.221:5202 timer:(on,204ms,0) ino:65753197 sk:f003 cgroup:/user.slice/user-0.slice/session-3640.scope <->
text = "ESTAB 0      2364360 10.22.116.220:33024 10.22.116.221:5202 timer:(on,204ms,0) ino:65753197 sk:f003 cgroup:/user.slice/user-0.slice/session-3640.scope ts sack ecn bbr wscale:7,7 rto:208 rtt:7.817/5.621 mss:4148 pmtu:4200 rcvmss:536 advmss:4148 cwnd:78 ssthresh:46 bytes_sent:48925697 bytes_retrans:9200264 bytes_acked:38543254 segs_out:11803 segs_in:5189 data_segs_out:11801 bbr:(bw:62.9Mbps,mrtt:0.017,pacing_gain:1,cwnd_gain:2) send 29.7Mbps lastrcv:6012 pacing_rate 62.3Mbps delivery_rate 27.5Mbps delivered:9508 busy:5968ms rwnd_limited:196ms(3.3%) unacked:285 retrans:0/2218 lost:77 sacked:201 dsack_dups:8 reordering:110 reord_seen:1888 rcv_space:41480 rcv_ssthresh:61388 notsent:1182180 minrtt:0.017"

#text = "cwnd:7 ssthresh:46 bytes_sent:48925697 bytes_retrans:9200264 bytes_acked:38543254 segs_out:11803 segs_in:5189 data_segs_out:11801 bbr:(bw:40.3Mbps,mrtt:0.017,pacing_gain:1,cwnd_gain:2) send 45.7Mbps lastsnd:4 lastrcv:1883256 lastack:4 pacing_rate 39.9Mbps delivery_rate 3.07Mbps"
match = re.match(r'.*cwnd:\d+.*', text)

if match:
    print(f"Matched: {match.group()}")
    print(f"Matched: {match.group(0)}")
    match = re.match(r'.*(cwnd:\d+).ssthresh*', text)
    if match is not None:
         print('cwnd {}'.format(match.group(1)))
    match = re.match(r'.*bbr:\(bw:(\S+).bps.*mrtt:(\S+),pac.*(pacing_rate \S+).bps.*(delivery_rate \S+).bps.*', text)
    if match is not None:
        csvformat='btl_bw {} | mrtt {} | {} | {}'.format(match.group(1), match.group(2), match.group(3), match.group(4))
        print(csvformat)
else:
    print("No match found")
