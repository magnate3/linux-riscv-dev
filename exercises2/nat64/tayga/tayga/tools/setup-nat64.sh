#!/bin/sh

if [ "$1" = -a ]; then
	ipv6_subnet="64:ff9b:0:0::/64"
	ipv4_subnet="10.40.0.0/16"
elif [ $# -ge 2 ]; then
	ipv6_subnet="$1"
	ipv4_subnet="$2"
	shift 2
	more_params="$*"
else
	cat <<EOF
Usage:
  setup-nat64.sh <IPv6 subnet> <IPv4 subnet>    setup with specified prefixes
  setup-nat64.sh -a                             use default setup
Note:
  IPv6 subnet must be /64, IPv4 subnet must be /16
Example:
  setup-nat64.sh 2001:db8:0:0::/64 10.10.0.0/16
EOF
	exit 1
fi

ipv6_prefix=`expr "$ipv6_subnet" : '\([^:]\+:[^:]\+:[^:]\+:[^:]\+\)::\/64$'`
if [ -z "$ipv6_prefix" ]; then
	echo "*** Invalid IPv6 subnet: $ipv6_subnet" >&2
	exit 1
fi

ipv4_prefix=`expr "$ipv4_subnet" : '\([0-9]\+\.[0-9]\+\)\.0\.0\/16$'`
if [ -z "$ipv4_prefix" ]; then
	echo "*** Invalid IPv4 subnet: $ipv4_subnet" >&2
	exit 1
fi

rmmod tayga 2>/dev/null

if [ -f tayga.ko ]; then
	module=tayga.ko
else
	module=tayga
fi

set -x

insmod $module ipv6_addr=$ipv6_prefix:0:ffff:0:2 ipv4_addr=$ipv4_prefix.255.2 \
	prefix=$ipv6_prefix::/96 dynamic_pool=$ipv4_prefix.0.0/17 \
	$more_params || exit 1

ip link set nat64 up
ip addr add $ipv6_prefix:0:ffff:0:1/64 dev nat64
ip addr add $ipv4_prefix.255.1/16 dev nat64

