#!/bin/bash
# Inject page to specific cache.
# Author: Maxim Menshchikov (MaximMenshchikov@gmail.com)
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

if [[ $1 == "" ]] ; then
	echo "Usage:"
	echo "$0 cachename [number of pages]"
	exit 1
fi

pages=${2:-1}

for i in `seq 1 $pages` ; do
	res=$(cat /proc/slabinfo | grep "^$1 " | head -1)
	if [[ -z "${res// }" ]] ; then
		echo "Cache not found: $1"
		exit 1
	fi
	array=($res)
	insmod slab_inject.ko 				\
		   	"slab=$1" 					\
			"active_objs=${array[1]}" 	\
			"num_objs=${array[2]}"
	if [[ $? == 0 ]] ; then
		rmmod slab_inject
	fi
done
exit $?
