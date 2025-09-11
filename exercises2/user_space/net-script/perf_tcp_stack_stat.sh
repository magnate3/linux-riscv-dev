#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <PID> [duration]"
    echo "Example: $0 12345 10"
    exit 1
fi

PID=$1
DURATION=${2:-10}

echo "📊 Running perf stat for PID=$PID over $DURATION seconds..."
echo

perf stat -a -r 3 -p "$PID" -e \
    syscalls:sys_enter_sendto,\
    syscalls:sys_enter_sendmsg,\
    syscalls:sys_enter_write,\
    syscalls:sys_enter_recvfrom,\
    syscalls:sys_enter_recvmsg,\
    syscalls:sys_enter_read,\
    net:net_dev_queue,\
    net:netif_receive_skb,\
    net:netif_rx,\
    net:tcp_send_reset,\
    net:tcp_retransmit_skb,\
    net:tcp_cleanup_rbuf,\
    net:tcp_recvmsg,\
    net:tcp_sendmsg,\
    net:skb_copy_datagram_iter,\
    napi:napi_poll,\
    context-switches,\
    migrations,\
    cpu-clock,\
    cycles,instructions,cache-references,cache-misses,mem-loads \
    -- sleep "$DURATION"
