#!/bin/bash

abip=$(ip addr show eth0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)
bcip=$(ip addr show eth1 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)

datasocket="${abip}:4433"
clientsocket="${bcip}:1234"
rustlog="error"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -srv-t) servertype="$2"; shift ;;
        -srv-pref) server_network_preference="$2"; shift ;;
        -srv-v) server_network_value="$2"; shift ;;
        -reliably) reliably="$2"; shift ;;
        -prio) prio="$2"; shift ;;
        -cc) cc="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Read the type of network tweaking
if [ -z "$servertype" ]
then
    type="static"
fi

# Read the network preference (delay, loss, rate)
if [ -z "$server_network_preference" ]
then
    server_network_preference="delay"
fi

# Read the value of delay, loss, or rate
if [ -z "$server_network_value" ]
then
    server_network_value="0"
fi

# Read if we like to send reliably 
if [ -z "$reliably" ]
then
    reliably="false"
fi

# Read the prioritization algorithm
if [ -z "$prio" ]
then
    prio="dynamic weighted"
fi

# Read the CC algorithm
if [ -z "$cc" ]
then
    cc="bbr"
fi

if [ "$type" = "default" ]
then
    tc qdisc del dev eth1 root 2>/dev/null
fi

# We use them mutual exclusive thus this small case distiction is okay. Otherwise this would grow large.
# Static means we use the same settings throughout the complete experiment
if [ "$type" = "static" ]
then
    if [ "$server_network_preference" = "delay" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem delay "${server_network_value}ms"
    fi

    if [ "$server_network_preference" = "loss" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem loss "${server_network_value}%"
    fi

    if [ "$server_network_preference" = "rate" ]
    then
        tc qdisc add dev eth1 root handle 1: tbf rate "${server_network_value}" burst 32kbit latency 400ms
    fi

    if [ "$server_network_preference" = "rateplus" ]
    then
        tc qdisc add dev eth1 root handle 1: tbf rate "${server_network_value}" burst 32kbit latency 400ms
    fi
fi

# Dynamic means that the parameter is interpreted as true or false
# In this mode we change the network settings after a few milliseconds
if [ "$type" = "dynamic" ]
then
    if [ "$server_network_preference" = "delay" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem delay "50ms"
    fi

    if [ "$server_network_preference" = "loss" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem loss "0%"
    fi

    if [ "$server_network_preference" = "rate" ]
    then
        tc qdisc add dev eth1 handle 1: tbf rate "100mbit" burst 32kbit latency 400ms
    fi
fi

rate=${server_network_value%m*}
sending_rate=$((${rate}*1000000))
# Clear any existing qdiscs
# tc qdisc del dev eth1 root 2>/dev/null

# Delay (here 100ms) and packet loss (here 5%)
# tc qdisc add dev eth1 root handle 1: netem delay 100ms loss 5%
# tc qdisc ad dev eth1 root handle 1: netem delay 100ms

# Bandwidth limiting (here 10mbit) burst and latency are recommended parameters
# If only using bandwidth limitation without delay or loss it is necessary to replace "parent 1: handle 10:" by "handle 1:""
# tc qdisc add dev eth1 parent 1: handle 10: tbf rate 10mbit burst 32kbit latency 400ms
if [ "$server_network_preference" = "rateplus" ]
then
    if [ "$reliably" = "true" ]
    then
        RUST_LOG="$rustlog" ./quiche-server --listen-from "${datasocket}" --listen-to "${clientsocket}" --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key --store-eval "./logs" --cc-algorithm "${cc}" --reliability --prio "static Content-type" --rate "${sending_rate}"
    else
        RUST_LOG="$rustlog" ./quiche-server --listen-from "${datasocket}" --listen-to "${clientsocket}" --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key --store-eval "./logs" --cc-algorithm "${cc}" --prio "${prio}" --rate "${sending_rate}"
    fi
else
    if [ "$reliably" = "true" ]
    then
        RUST_LOG="$rustlog" ./quiche-server --listen-from "${datasocket}" --listen-to "${clientsocket}" --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key --store-eval "./logs" --cc-algorithm "${cc}" --reliability --prio "static Content-type"
    else
        RUST_LOG="$rustlog" ./quiche-server --listen-from "${datasocket}" --listen-to "${clientsocket}" --cert apps/src/bin/cert.crt --key apps/src/bin/cert.key --store-eval "./logs" --cc-algorithm "${cc}" --prio "${prio}"
    fi
fi

if [ "$type" = "dynamic" ]
then
    sleep 5
    if [ "$server_network_preference" = "delay" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem delay "100ms"
        sleep 5
        tc qdisc ad dev eth1 root handle 1: netem delay "10ms"
        sleep 5
        tc qdisc ad dev eth1 root handle 1: netem delay "50ms"
    fi

    if [ "$server_network_preference" = "loss" ]
    then
        tc qdisc ad dev eth1 root handle 1: netem loss "2%"
        sleep 5
        tc qdisc ad dev eth1 root handle 1: netem loss "1%"
        sleep 5
        tc qdisc ad dev eth1 root handle 1: netem loss "0%"
    fi

    if [ "$server_network_preference" = "rate" ]
    then
        tc qdisc add dev eth1 handle 1: tbf rate "10mbit" burst 32kbit latency 400ms
        sleep 5
        tc qdisc add dev eth1 handle 1: tbf rate "1gbit" burst 32kbit latency 400ms
        sleep 5
        tc qdisc add dev eth1 handle 1: tbf rate "100mbit" burst 32kbit latency 400ms
    fi
fi