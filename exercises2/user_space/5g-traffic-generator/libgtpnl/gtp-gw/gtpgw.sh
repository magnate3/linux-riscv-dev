#!/bin/bash

# GTP-link usage: Usage: %s add <device> <family> [--sgsn] [<address>]
# [--sgsn] [<address>] are optional

## Configuration

# Can be: ip, ip6
IP_VERSION="ip"

NS1=100
NS2=200

NS1_NAME="ns-gtp-$NS1"
NS2_NAME="ns-gtp-$NS2"

TEID_OUT="100"
TEID_IN="200"

GTP_DEV1="gtp-$NS1"
GTP_DEV2="gtp-$NS2"

VETH_DEV1="veth-$NS1"
VETH_DEV2="veth-$NS2"

NS1_LO_IP="192.168.0.$TEID_OUT"
NS1_LO_CIDR="$NS1_LO_IP/32"
NS2_LO_IP="192.168.0.$TEID_IN"
NS2_LO_CIDR="$NS2_LO_IP/32"

NS1_VETH_IP="172.100.0.$TEID_OUT"
NS1_VETH_CIDR="$NS1_VETH_IP/24"
NS2_VETH_IP="172.100.0.$TEID_IN"
NS2_VETH_CIDR="$NS2_VETH_IP/24"

NS1_EXEC="ip netns exec $NS1_NAME"
NS2_EXEC="ip netns exec $NS2_NAME"

## Logger
function log {
  echo -e "[INFO] $1"
  sleep 1
}

function print_test_msg {
  echo ""
  echo "You can do for e.g.:"
  echo "  $NS1_EXEC ping $NS2_LO_IP"
  echo "  $NS2_EXEC ping $NS1_LO_IP"
  echo ""
  echo "Using tshark you will see ICMP pckets encapsulated in GTP"
  echo ""
}

## Create veth pairs, gtp namespaces and ifaces
function start {
  log "create veth pairs"
  ip link add $VETH_DEV1 type veth peer name $VETH_DEV2

  log "create network namespaces"
  ip netns add $NS1_NAME
  ip netns add $NS2_NAME

  log "attribute each veth pair to its correspondent netns"
  ip link set $VETH_DEV1 netns $NS1_NAME
  ip link set $VETH_DEV2 netns $NS2_NAME

  log "set ip addresses of veth pairs and loopbacks"
  $NS1_EXEC ip addr add $NS1_VETH_CIDR dev $VETH_DEV1
  $NS1_EXEC ip addr add $NS1_LO_CIDR dev lo
  $NS2_EXEC ip addr add $NS2_VETH_CIDR dev $VETH_DEV2
  $NS2_EXEC ip addr add $NS2_LO_CIDR dev lo

  log "enable veth and lo interfaces"
  $NS1_EXEC ip link set $VETH_DEV1 up
  $NS1_EXEC ip link set lo up
  $NS2_EXEC ip link set $VETH_DEV2 up
  $NS2_EXEC ip link set lo up

  log "create gtp devices (run in bg mode)"
  $NS1_EXEC ./gtp-link add $GTP_DEV1 $IP_VERSION &
  $NS2_EXEC ./gtp-link add $GTP_DEV2 $IP_VERSION &

  log "configure mtu of gtp devices"
  $NS1_EXEC ifconfig $GTP_DEV1 mtu 1500 up
  $NS2_EXEC ifconfig $GTP_DEV2 mtu 1500 up

  log "create gtp tunnels"
  $NS1_EXEC ./gtp-tunnel add $GTP_DEV1 v1 $TEID_IN $TEID_OUT $NS2_LO_IP $NS2_VETH_IP
  log "$($NS1_EXEC ./gtp-tunnel list)"
  $NS2_EXEC ./gtp-tunnel add $GTP_DEV2 v1 $TEID_OUT $TEID_IN $NS1_LO_IP $NS1_VETH_IP
  log "$($NS2_EXEC ./gtp-tunnel list)"

  log "configure routes using gtp devices"
  $NS1_EXEC ip route add $NS2_LO_CIDR dev $GTP_DEV1
  $NS2_EXEC ip route add $NS1_LO_CIDR dev $GTP_DEV2

  print_test_msg
}

## Destroy everything
function stop {
  log "remove gtp devices"
  $NS1_EXEC ./gtp-link del $GTP_DEV1
  $NS2_EXEC ./gtp-link del $GTP_DEV2

  log "remove network namespaces"
  ip netns del $NS1_NAME
  ip netns del $NS2_NAME
}

if [ "$1" = "start" ]; then
  start
elif [ "$1" = "stop" ]; then
  stop
else
  echo "This is an example to create gtp tunnel and send some traffic"
  echo ""
  echo "  Usage: $0 <start|stop>"
  echo ""
fi

