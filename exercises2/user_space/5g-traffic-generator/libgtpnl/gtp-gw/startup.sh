#!/bin/sh

set -e

_authors="abousselmi and sofianinho"
_license="LGPLv2"
_version="v1.0.0"

IP_VERSION="ip"
MTU=1500
GTP_DEV="ue-tun"

banner() {
    figlet -f big "GTP-GW"
    echo "version: $_version"
    echo "authors: $_authors"
    echo "license: $_license"
    echo ""
}

log() {
    echo -e "$(date +%F-%T) [INFO] $1"
}

checks() {
    if [ -z "$UE_IP" ]; then echo "UE_IP is not set.." && exit 1 ; fi
    if [ -z "$N3_GNB_IP" ]; then echo "N3_GNB_IP is not set.." && exit 1 ; fi
    if [ -z "$N3_UPF_IP" ]; then echo "N3_UPF_IP is not set.." && exit 1 ; fi
    if [ -z "$TEID_N3_UPF" ]; then echo "TEID_N3_UPF is not set.." && exit 1 ; fi
    if [ -z "$TEID_N3_GNB" ]; then echo "TEID_N3_GNB is not set.." && exit 1 ; fi
    if [ -z "$N6_APP_SERVER_IP" ]; then echo "N6_APP_SERVER_IP is not set.." && exit 1 ; fi
}

start() {
    log "starting gtp-gw.."

    log "add UE ip [$UE_IP] to loopback interface"
    ip addr add $UE_IP/32 dev lo

    log "create [$GTP_DEV] gtp device"
    gtp-link add $GTP_DEV $IP_VERSION > /dev/null 2>&1 &

    echo $! > pid

    sleep 1

    log "set [$GTP_DEV] mtu to [$MTU]"
    ip link set mtu $MTU dev $GTP_DEV

    log "create gtp tunnel"
    gtp-tunnel add $GTP_DEV v1 $TEID_N3_UPF $TEID_N3_GNB $N6_APP_SERVER_IP $N3_UPF_IP

    sleep 1

    log "configure routes using gtp devices"
    ip route add $N6_APP_SERVER_IP/32 dev $GTP_DEV

    log "------------------------------------------------------------------------"
    log "UE if: $GTP_DEV"
    log "UE ip: $UE_IP"
    log "APP server ip: $N6_APP_SERVER_IP"
    log "N3 GNB ip: $N3_GNB_IP"
    log "N3 UPF ip: $N3_UPF_IP"
    log "GNB GTP TEID: $TEID_N3_GNB"
    log "UPF GTP TEID: $TEID_N3_UPF"
    log "------------------------------------------------------------------------"
    log "$(gtp-tunnel list)"
    log "------------------------------------------------------------------------"
    log "You can run: ping $N6_APP_SERVER_IP"
}

stop() {
    log "stopping gtp-gw.."

    log "delete loopback UE IP config"
    ip addr del $UE_IP/32 dev lo

    log "delete gtp-tunnel"
    gtp-tunnel del $GTP_DEV v1 $TEID_N3_UPF $IP_VERSION

    log "delete the gtp device"
    gtp-link del $GTP_DEV

    log "kill the gtp device process"
    kill -9 $(cat pid)
}

usage() {
    echo ""
    echo "Usage: $0 <start|stop>"
    echo ""
}

case "$1" in
    -h | --help | help)
        usage
	exit
	;;
    start | add | create)
	banner
	checks
	start
	;;
    stop | delete | del)
	checks
	stop
	;;
    *)
	usage
	exit
	;;
esac
