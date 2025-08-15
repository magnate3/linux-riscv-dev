#!/bin/sh

if [ -e gogtp5g-link.pid ]; then
	echo "===> Shutdown gogtp5g-link"
	kill `cat gogtp5g-link.pid`
fi
ip netns exec CL ./gtp5g-link del gtp5gtest
ip netns del CL
ip netns del SV

echo "===> Unbind"
#driverctl unset-override 0000:00:11.0
#driverctl unset-override 0000:00:09.0
#driverctl list-overrides
