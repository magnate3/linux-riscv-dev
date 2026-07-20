#!/usr/bin/env python3
import subprocess
import shlex
import time
import re
import sys


def pp(obj):
    from pprint import pprint
    pprint(obj, width=180)


items = [
    [
        ['algo', r'^([^ ]+)'],
        ['wscale', r':(\d+,\d+)'],
        ['backoff', r':(\d+)'],
        ['rto', r':(\d+)'],
        ['rtt', r':([0-9.]+/[0-9.]+)'],
        ['ato', r':(\d+)'],
        ['mss', r':(\d+)'],
        ['pmtu', r':(\d+)'],
        ['rcvmss', r':(\d+)'],
        ['advmss', r':(\d+)'],
        ['cwnd', r':(\d+)'],
        ['ssthresh', r':(\d+)'],
    ],
    [
        ['bytes_sent', r':(\d+)'],
        ['bytes_acked', r':(\d+)'],
        ['bytes_received', r':(\d+)'],
        ['segs_out', r':(\d+)'],
        ['segs_in', r':(\d+)'],
        ['data_segs_out', r':(\d+)'],
        ['data_segs_in', r':(\d+)'],
        ['send', r' ([0-9.]+\wbps)'],
        ['lastsnd', r':(\d+)'],
        ['lastrcv', r':(\d+)'],
        ['lastack', r':(\d+)'],
    ],
    [
        ['pacing_rate', r' ([0-9.]+\wbps)'],
        ['delivery_rate', r' ([0-9.]+\wbps)'],
        ['delivered', r':(\d+)'],
        ['busy', r':(\d+)ms'],
        ['retrans', r':(\d+/\d+)'],
        ['rcv_rtt', r':([0-9.]+)'],
        ['rcv_space', r':(\d+)'],
        ['rcv_ssthresh', r':(\d+)'],
        ['minrtt', r':([0-9.]+)'],
    ]
]


def parse_ssit(data):
    lines = data.splitlines(True)[1:]
    lines = list(filter(None, lines))

    result = []
    for i in range(0, len(lines), 2):
        a = lines[i]
        b = lines[i + 1].strip()
        ret = {}

        m = re.match(r"(?P<state>[^ ]+) +(?P<recvq>[^ ]+) +(?P<sendq>[^ ]+) +(?P<localaddr>[^ ]+) +(?P<peeraddr>[^ ]+)(?: +(?P<process>.+))?", a)
        ret['conn'] = dict(
            state=m.group('state'),
            recvq=m.group('recvq'),
            sendq=m.group('sendq'),
            localaddr=m.group('localaddr'),
            peeraddr=m.group('peeraddr'),
            process=m.group('process'),
        )
        ret['stat'] = {}
        for tmp in items:
            for item in tmp:
                key = item[0]
                pat = key + item[1] if key != "algo" else item[1]
                m = re.search(pat, b)
                if m:
                    ret['stat'][key] = m.group(1)
                else:
                    ret['stat'][key] = "0"

        result.append(ret)
    return result


def output_ssit(result):
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        BG = '\033[7;37;40m'


    def outout_entry(result, colored=False):
        for i, tmp in enumerate(items):
            if colored:
                print(bcolors.BG, end='')
            if i == 0:
                print('{:<21} {:<21}'.format(res['conn']['localaddr'], res['conn']['peeraddr']), end='')
            if i == 1:
                print('{:<10} {:>15} {:>14}'.format(res['conn']['state'], res['conn']['recvq'], res['conn']['sendq']), end='')
            else:
                print('{:<21} {:<21}'.format('', ''), end='')

            for item in tmp:
                key = item[0]
                print(' {key:>{width}}'.format(key=res['stat'][key], width=14), end='')
            if colored:
                print(bcolors.ENDC, end='')
            print('')

    # output header
    res = dict(
        conn=dict(
            localaddr='local addr',
            peeraddr='peer addr',
            state='State',
            recvq='Recv-Q',
            sendq='Send-Q',
        ),
        stat=dict(),
    )
    for tmp in items:
        for item in tmp:
            key = item[0]
            res['stat'][key] = key
    outout_entry(res, colored=True)
    for j, res in enumerate(result):
        colored = True if j % 2 else False
        outout_entry(res, colored)


def main():
    data = """State        Recv-Q   Send-Q      Local Address:Port       Peer Address:Port    Process                                                                         
ESTAB        0        0               127.0.0.1:58140         127.0.0.1:46277    users:(("sshd",pid=35747,fd=10))
    cubic wscale:7,7 rto:212 rtt:9.102/15.24 ato:40 mss:65483 pmtu:65535 rcvmss:780 advmss:65483 cwnd:10 ssthresh:22 bytes_sent:1477787 bytes_retrans:22340 bytes_acked:1455448 bytes_received:5741479 segs_out:972 segs_in:775 data_segs_out:448 data_segs_in:552 send 575.5Mbps lastsnd:2220 lastrcv:4228 lastack:2176 pacing_rate 1151.1Mbps delivery_rate 96501.3Mbps delivered:449 busy:1996ms retrans:0/1 dsack_dups:1 rcv_rtt:19.627 rcv_space:632400 rcv_ssthresh:3144401 minrtt:0.008
ESTAB        0        0               127.0.0.1:46277         127.0.0.1:58142    users:(("node",pid=35981,fd=19))
    cubic wscale:7,7 rto:208 rtt:4.865/8.786 ato:40 mss:65483 pmtu:65535 rcvmss:1164 advmss:65483 cwnd:10 bytes_sent:6651738 bytes_acked:6651738 bytes_received:6261988 segs_out:1894 segs_in:2532 data_segs_out:1252 data_segs_in:1426 send 1076.8Mbps lastsnd:32 lastrcv:32 lastack:32 pacing_rate 2153.5Mbps delivery_rate 23812.0Mbps delivered:1253 app_limited busy:5148ms rcv_rtt:10.126 rcv_space:327680 rcv_ssthresh:3144299 minrtt:0.005
ESTAB        0        0               127.0.0.1:46277         127.0.0.1:58140    users:(("node",pid=35787,fd=18))
    cubic wscale:7,7 rto:204 rtt:2.661/5.16 ato:40 mss:65483 pmtu:65535 rcvmss:22340 advmss:65483 cwnd:10 bytes_sent:5741479 bytes_acked:5741479 bytes_received:1455447 segs_out:774 segs_in:972 data_segs_out:552 data_segs_in:448 send 1968.7Mbps lastsnd:4228 lastrcv:2220 lastack:2220 pacing_rate 3936.8Mbps delivery_rate 87310.7Mbps delivered:553 busy:2008ms rwnd_limited:4ms(0.2%) rcv_rtt:4.836 rcv_space:129024 rcv_ssthresh:2815203 minrtt:0.007
ESTAB        0        0               127.0.0.1:58142         127.0.0.1:46277    users:(("sshd",pid=35747,fd=11))
    cubic wscale:7,7 rto:208 rtt:5.794/10.362 ato:40 mss:65483 pmtu:65535 rcvmss:65483 advmss:65483 cwnd:10 ssthresh:17 bytes_sent:6286957 bytes_retrans:24969 bytes_acked:6261989 bytes_received:6651738 segs_out:2532 segs_in:1895 data_segs_out:1426 data_segs_in:1252 send 904.1Mbps lastsnd:32 lastrcv:32 lastack:32 pacing_rate 1084.9Mbps delivery_rate 104772.8Mbps delivered:1427 app_limited busy:6072ms retrans:0/2 dsack_dups:2 rcv_rtt:13.784 rcv_space:409600 rcv_ssthresh:3144366 minrtt:0.005
CLOSE-WAIT   32       0          192.168.61.130:34020        8.43.85.13:https    users:(("gnome-shell",pid=1662,fd=27))
    cubic rto:488 rtt:103.808/95.179 ato:40 mss:1460 pmtu:1500 rcvmss:1460 advmss:1460 cwnd:10 bytes_sent:2033 bytes_acked:2034 bytes_received:4516 segs_out:10 segs_in:11 data_segs_out:4 data_segs_in:6 send 1.1Mbps lastsnd:76041660 lastrcv:76011444 lastack:76011444 pacing_rate 2.3Mbps delivery_rate 40.4Mbps delivered:5 app_limited rcv_space:14600 rcv_ssthresh:64076 minrtt:0.289
ESTAB        0        0          192.168.61.130:ssh        192.168.61.1:35818    users:(("sshd",pid=35747,fd=4),("sshd",pid=35667,fd=4))
    cubic wscale:14,7 rto:216 rtt:14.627/13.319 ato:40 mss:1460 pmtu:1500 rcvmss:1204 advmss:1460 cwnd:10 ssthresh:81 bytes_sent:12488592 bytes_acked:12488592 bytes_received:7811233 segs_out:11476 segs_in:8975 data_segs_out:10277 data_segs_in:6954 send 8.0Mbps lastsnd:32 lastrcv:32 lastack:32 pacing_rate 16.0Mbps delivery_rate 332.7Mbps delivered:10278 busy:31280ms rcv_rtt:5.359 rcv_space:425984 rcv_ssthresh:3144640 minrtt:0.079
"""  # noqa: E501

    option = ' '.join(sys.argv[1:])
    while True:
        data = subprocess.check_output(shlex.split(f"ss -itp {option}")).decode('utf-8')
        result = parse_ssit(data)
        output_ssit(result)
        time.sleep(1)


if __name__ == "__main__":
    main()
